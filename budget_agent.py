"""Агент семейного бюджета: SGR + RAG + MCP + Telegram."""
import datetime as _dt
import json
import logging
import math
import os
import re
import select
import shlex
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from typing import Annotated, Literal, Union

import httpx
import psycopg
from dotenv import load_dotenv
from openai import OpenAI
from pgvector.psycopg import register_vector
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, ValidationError, model_validator

load_dotenv()

# ===== 1. КОНФИГУРАЦИЯ =====

@dataclass(frozen=True)
class Settings:
    provider: str
    ollama_url: str
    ollama_model: str
    router_url: str | None
    router_key: str | None
    router_model: str | None
    structured_mode: str
    disable_thinking: bool
    embed_url: str
    embed_model: str
    pg_dsn: str
    telegram_token: str | None
    telegram_proxy: str | None
    husband_tg_id: int | None
    wife_tg_id: int | None
    allowed_group_chat_ids: tuple[int, ...]
    mcp_cbr_cmd: str
    mcp_init_timeout_seconds: float
    mcp_call_timeout_seconds: float
    broadcast_steps: bool

def load_settings() -> Settings:
    def _int(name):
        v = os.getenv(name, "").strip()
        return int(v) if v else None
    def _float(name, default):
        v = os.getenv(name, "").strip()
        return float(v) if v else default
    def _need(name):
        v = os.getenv(name, "").strip()
        if not v:
            sys.exit(f"Ошибка: не задана {name}. Определите её в .env или в окружении.")
        return v
    def _int_tuple(name):
        raw = os.getenv(name, "").strip()
        return tuple(int(v.strip()) for v in raw.split(",") if v.strip())

    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    if provider == "ollama":
        ollama_url = _need("OLLAMA_URL").rstrip("/")
        ollama_model = _need("OLLAMA_MODEL")
    else:
        # OLLAMA_URL/OLLAMA_MODEL нужны только в режиме ollama; в router их отсутствие не ошибка.
        ollama_url = os.getenv("OLLAMA_URL", "").strip().rstrip("/")
        ollama_model = os.getenv("OLLAMA_MODEL", "").strip()

    s = Settings(
        provider=provider,
        ollama_url=ollama_url,
        ollama_model=ollama_model,
        router_url=os.getenv("ROUTER_API_URL"),
        router_key=os.getenv("ROUTER_API_KEY") or os.getenv("AGENT_PLATFORM_API_KEY"),
        router_model=os.getenv("ROUTER_MODEL"),
        structured_mode=os.getenv("LLM_STRUCTURED_MODE", "auto").strip().lower(),
        disable_thinking=os.getenv("LLM_DISABLE_THINKING", "1") == "1",
        # Эмбеддинги всегда идут через локальную Ollama, независимо от LLM_PROVIDER.
        embed_url=_need("EMBED_URL").rstrip("/"),
        embed_model=_need("EMBED_MODEL"),
        pg_dsn=os.getenv("PG_DSN", ""),
        telegram_token=os.getenv("TELEGRAM_BOT_TOKEN"),
        telegram_proxy=os.getenv("TELEGRAM_HTTP_PROXY") or os.getenv("HTTPS_PROXY"),
        husband_tg_id=_int("PERSON_HUSBAND_TG_ID"),
        wife_tg_id=_int("PERSON_WIFE_TG_ID"),
        allowed_group_chat_ids=_int_tuple("TELEGRAM_ALLOWED_GROUP_CHAT_IDS"),
        mcp_cbr_cmd=_need("MCP_CBR_CMD"),
        # Первый запуск через uvx (скачивание/распаковка пакета) заметно дольше
        # обычного вызова — поэтому у инициализации свой, более щедрый таймаут.
        mcp_init_timeout_seconds=_float("MCP_INIT_TIMEOUT_SECONDS", 60.0),
        mcp_call_timeout_seconds=_float("MCP_CALL_TIMEOUT_SECONDS", 15.0),
        broadcast_steps=os.getenv("BROADCAST_STEPS", "1") == "1",
    )
    if not s.pg_dsn:
        sys.exit("Ошибка: не задан PG_DSN.")
    if s.provider == "router" and not (s.router_url and s.router_key and s.router_model):
        sys.exit("Ошибка: для LLM_PROVIDER=router нужны ROUTER_API_URL, ROUTER_API_KEY, ROUTER_MODEL.")
    return s

# --doctor обязан отработать даже при полностью отсутствующем/неполном .env —
# это ровно тот случай, который он должен диагностировать до восьминутной
# индексации. load_settings() завершает процесс через sys.exit() при нехватке
# обязательной переменной, поэтому на уровне модуля его вызов в режиме
# доктора обходим стороной: cmd_doctor() читает окружение напрямую (раздел
# 11) и в SETTINGS не заглядывает. Поведение всех остальных режимов не
# меняется — они по-прежнему проходят через load_settings() и падают с
# тем же понятным сообщением, что и раньше.
if "--doctor" in sys.argv[1:]:
    SETTINGS = None
else:
    SETTINGS = load_settings()   # единственный экземпляр, используется всеми разделами

# ===== 2. СЛОЙ LLM =====

class StructuredFailure(RuntimeError):
    """Модель не вернула разбираемый ответ, соответствующий схеме."""

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)

def extract_json(text: str) -> dict:
    """Снимает markdown-обёртку и выделяет внешний объект. Типы НЕ приводятся."""
    if not text or not text.strip():
        raise StructuredFailure("пустой ответ модели")
    candidate = text.strip()
    m = _FENCE.search(candidate)
    if m:
        candidate = m.group(1).strip()
    else:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            raise StructuredFailure(f"JSON не найден: {candidate[:120]!r}")
        candidate = candidate[start:end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as e:
        raise StructuredFailure(f"невалидный JSON: {e}") from e
    if not isinstance(parsed, dict):
        raise StructuredFailure("ожидался объект JSON")
    return parsed

_client: OpenAI | None = None

def llm_client() -> OpenAI:
    global _client
    if _client is None:
        s = SETTINGS
        if s.provider == "ollama":
            _client = OpenAI(base_url=f"{s.ollama_url}/v1", api_key="ollama")
        else:
            _client = OpenAI(base_url=s.router_url, api_key=s.router_key)
    return _client

def _model_name() -> str:
    return SETTINGS.ollama_model if SETTINGS.provider == "ollama" else SETTINGS.router_model

def _raw_completion(messages, response_format=None, max_tokens=1200):
    kwargs = {"model": _model_name(), "messages": messages, "max_tokens": max_tokens}
    if response_format:
        kwargs["response_format"] = response_format
    extra = {}
    if SETTINGS.disable_thinking:
        extra["thinking"] = {"type": "disabled"}   # DeepSeek V4
        extra["think"] = False                      # Ollama
    if extra:
        kwargs["extra_body"] = extra
    resp = llm_client().chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""

def _schema_prompt(schema: type[BaseModel]) -> str:
    js = json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2)
    return ("Ответь строго одним объектом json по схеме ниже. "
            "Только json, без пояснений и без markdown-обёртки.\n\n" + js)

def structured_call(schema, messages, mode=None, max_tokens=1200):
    mode = mode or SETTINGS.structured_mode
    if mode == "auto":
        mode = detect_mode()
    convo = list(messages)
    if mode in ("json_object", "prompt"):
        convo = [{"role": "system", "content": _schema_prompt(schema)}] + convo
    rf = None
    if mode == "strict":
        rf = {"type": "json_schema", "json_schema": {
            "name": schema.__name__, "strict": True, "schema": schema.model_json_schema()}}
    elif mode == "json_object":
        rf = {"type": "json_object"}

    last_error = None
    for attempt in (1, 2):
        try:
            raw = _raw_completion(convo, response_format=rf, max_tokens=max_tokens)
            return schema.model_validate(extract_json(raw))
        except (StructuredFailure, ValidationError) as e:
            last_error = e
            if attempt == 2:
                break
            convo = convo + [{"role": "user", "content":
                f"Предыдущий ответ не подошёл: {e}. Верни json строго по схеме."}]
    raise StructuredFailure(f"два неудачных разбора подряд: {last_error}")

class _Probe(BaseModel):
    model_config = ConfigDict(extra="forbid")
    city: str
    temp_c: float
    sunny: bool

_PROBE_MSGS = [{"role": "user", "content": "Погода в Париже. Верни json."}]
_detected: str | None = None
_probe_results: dict[str, bool] = {}   # кэш: перебор кандидатов выполняется не более раза на режим

def _probe(mode: str) -> bool:
    if mode in _probe_results:
        return _probe_results[mode]
    try:
        obj = structured_call(_Probe, _PROBE_MSGS, mode=mode, max_tokens=300)
        ok = isinstance(obj, _Probe)
    except Exception:
        ok = False
    _probe_results[mode] = ok
    return ok

def detect_mode() -> str:
    """Сверяет фактические поля ответа со схемой — облачная Ollama принимает
    response_format, но молча игнорирует его, поэтому проверка идёт по данным,
    а не по факту принятия параметра API."""
    global _detected
    if _detected is None:
        for candidate in ("strict", "json_object", "prompt"):
            if _probe(candidate):
                _detected = candidate
                break
        else:
            _detected = "prompt"
    return _detected

# ===== 3. ЭМБЕДДИНГИ =====

EMBED_DIM = 1024

def embed(text: str) -> list[float]:
    r = httpx.post(f"{SETTINGS.embed_url}/api/embed",
                   json={"model": SETTINGS.embed_model, "input": text}, timeout=120)
    r.raise_for_status()
    vec = r.json()["embeddings"][0]
    if len(vec) != EMBED_DIM:
        raise RuntimeError(f"ожидалось {EMBED_DIM} измерений, пришло {len(vec)}")
    return [float(x) for x in vec]

# ===== 4. ХРАНИЛИЩЕ =====

def db() -> psycopg.Connection:
    conn = psycopg.connect(SETTINGS.pg_dsn, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn

SCOPES = ("common", "husband", "wife")

@dataclass(frozen=True)
class Ctx:
    person: str | None       # husband | wife | None
    chat_type: str           # private | group
    chat_id: int
    user_id: int | None = None

def resolve_person_id(conn, person: str | None) -> int | None:
    """Числовой id персоны по её роли ('husband'/'wife'). Не хранится в Ctx:
    persons.id — serial, растёт при каждой перезаливке сида, а роль нет —
    числовой id, пронесённый через Ctx с прошлого запроса, неизбежно
    отстаёт от реальной таблицы persons. Резолвится заново на каждый вызов."""
    if person not in ("husband", "wife"):
        return None
    row = conn.execute("SELECT id FROM persons WHERE role = %s", (person,)).fetchone()
    return row[0] if row else None

def visible_scopes(person: str | None, chat_type: str) -> tuple[str, ...]:
    """Личный чат: общее плюс своё. Любой другой chat_type — включая незнакомый,
    пустой или None — только общее: закрываемся по умолчанию (allow-list на
    "private"), а не открываемся по умолчанию (deny-list на "group" пропускал бы
    supergroup/channel/etc., которые реально приходят от Telegram)."""
    if chat_type == "private" and person in ("husband", "wife"):
        return ("common", person)
    return ("common",)

DDL = f"""
CREATE TABLE IF NOT EXISTS persons (
    id serial PRIMARY KEY, name text NOT NULL,
    role text NOT NULL CHECK (role IN ('husband','wife')),
    telegram_id bigint UNIQUE);

CREATE TABLE IF NOT EXISTS accounts (
    id serial PRIMARY KEY, name text NOT NULL,
    kind text NOT NULL CHECK (kind IN ('cash','card','deposit')),
    currency text NOT NULL CHECK (currency IN ('RUB','USD')),
    opening_balance numeric(14,2) NOT NULL DEFAULT 0,
    scope text NOT NULL CHECK (scope IN {SCOPES}));

CREATE TABLE IF NOT EXISTS categories (
    id serial PRIMARY KEY, name text UNIQUE NOT NULL,
    kind text NOT NULL CHECK (kind IN ('expense','income')));

CREATE TABLE IF NOT EXISTS transactions (
    id serial PRIMARY KEY, ts timestamptz NOT NULL DEFAULT now(),
    amount numeric(14,2) NOT NULL CHECK (amount > 0),
    currency text NOT NULL CHECK (currency IN ('RUB','USD')),
    account_id int REFERENCES accounts(id),
    category_id int REFERENCES categories(id),
    merchant text, comment text,
    scope text NOT NULL CHECK (scope IN {SCOPES}),
    person_id int REFERENCES persons(id));

CREATE TABLE IF NOT EXISTS recurring (
    id serial PRIMARY KEY, title text NOT NULL,
    amount numeric(14,2) NOT NULL,
    currency text NOT NULL CHECK (currency IN ('RUB','USD')),
    day_of_month int NOT NULL CHECK (day_of_month BETWEEN 1 AND 28),
    category_id int REFERENCES categories(id),
    scope text NOT NULL CHECK (scope IN {SCOPES}),
    active boolean NOT NULL DEFAULT true, note text);

CREATE TABLE IF NOT EXISTS goals (
    id serial PRIMARY KEY, title text NOT NULL,
    target_amount numeric(14,2) NOT NULL,
    currency text NOT NULL CHECK (currency IN ('RUB','USD')),
    due_date date, saved_amount numeric(14,2) NOT NULL DEFAULT 0,
    scope text NOT NULL CHECK (scope IN {SCOPES}));

CREATE TABLE IF NOT EXISTS merchant_aliases (
    alias text NOT NULL, category_id int NOT NULL REFERENCES categories(id),
    scope text NOT NULL DEFAULT 'common' CHECK (scope IN {SCOPES}),
    PRIMARY KEY (alias, scope));

CREATE TABLE IF NOT EXISTS family_rules (
    key text PRIMARY KEY, value_num numeric(14,2), value_text text,
    currency text, unit text, document_key text,
    scope text NOT NULL DEFAULT 'common' CHECK (scope IN {SCOPES}));

CREATE TABLE IF NOT EXISTS budget_limits (
    person_id int REFERENCES persons(id), period text NOT NULL,
    amount numeric(14,2) NOT NULL, currency text NOT NULL DEFAULT 'RUB',
    scope text NOT NULL DEFAULT 'common' CHECK (scope IN {SCOPES}),
    PRIMARY KEY (period, scope));

CREATE TABLE IF NOT EXISTS dialog_state (
    chat_id bigint NOT NULL, user_id bigint NOT NULL DEFAULT 0,
    person_id int REFERENCES persons(id),
    state text NOT NULL DEFAULT 'base', pending jsonb,
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (chat_id, user_id));

CREATE TABLE IF NOT EXISTS documents (
    id serial PRIMARY KEY, document_key text UNIQUE NOT NULL,
    doc_type text NOT NULL CHECK (doc_type IN ('knowledge','household')),
    title text NOT NULL, source text,
    scope text NOT NULL DEFAULT 'common' CHECK (scope IN {SCOPES}),
    text text NOT NULL, keywords text[], metadata jsonb,
    embedding vector({EMBED_DIM}));
"""

def create_schema(conn) -> None:
    conn.execute(DDL)


# Таблицы сид-данных в порядке, безопасном для DELETE (сначала те, у кого есть
# внешние ключи на остальные, последними — независимые/референсные).
_SEED_TABLES_DELETE_ORDER = (
    "transactions", "recurring", "budget_limits", "merchant_aliases",
    "goals", "accounts", "categories", "persons", "family_rules",
)

class SeedNotEmpty(RuntimeError):
    """load_seed вызван без force=True, а сид-таблицы уже содержат данные —
    среди них могут быть настоящие записи пользователя (бот пишет в эти же
    таблицы transactions/accounts/goals при разборе сообщений), поэтому без
    явного согласия перезаливка не выполняется."""

def load_seed(conn, path: str = "seed_data.json", force: bool = False) -> dict[str, int]:
    """Загружает сид-данные из JSON.

    Без force=True — защитный режим: если хоть одна из сид-таблиц не пуста,
    ничего не трогает и поднимает SeedNotEmpty. Таблицы transactions/accounts/
    goals — рабочие, бот пишет в них при разборе сообщений пользователя
    (задача 7), поэтому молчаливая перезаливка здесь недопустима: нечем
    отличить настоящую транзакцию от сид-овой.

    С force=True — полностью очищает свои таблицы и перезаливает (documents
    и dialog_state не трогает ни в каком режиме — их нет в seed_data.json).
    Так же безопасно вызывать повторно на непустой базе с force=True: это не
    дублирует транзакции и не падает на конфликте ключей, потому что заливка
    всегда начинается с чистых таблиц, а не с точечных upsert по естественным
    ключам, которых у этих таблиц просто нет (имя счёта не UNIQUE, у
    транзакции нет ключа кроме id)."""
    data = json.load(open(path, encoding="utf-8"))
    counts: dict[str, int] = {}
    with conn.transaction():
        if not force:
            for table in _SEED_TABLES_DELETE_ORDER:
                n = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                if n:
                    raise SeedNotEmpty(
                        f"таблица {table} уже содержит {n} строк(и) — база не пуста. "
                        "Повторная загрузка без явного согласия на перезапись отключена: "
                        "вызовите load_seed(conn, path, force=True), если действительно "
                        "нужно стереть текущие данные и перезалить сид."
                    )
        for table in _SEED_TABLES_DELETE_ORDER:
            conn.execute(f"DELETE FROM {table}")

        for p in data["persons"]:
            conn.execute("INSERT INTO persons (name, role, telegram_id) VALUES (%s,%s,%s)",
                         (p["name"], p["role"], p["telegram_id"]))
        counts["persons"] = len(data["persons"])
        person_id = {role: i for i, role in conn.execute("SELECT id, role FROM persons").fetchall()}

        for c in data["categories"]:
            conn.execute("INSERT INTO categories (name, kind) VALUES (%s,%s)",
                         (c["name"], c["kind"]))
        counts["categories"] = len(data["categories"])
        cat_id = {n: i for i, n in conn.execute("SELECT id, name FROM categories").fetchall()}

        for a in data["accounts"]:
            conn.execute(
                "INSERT INTO accounts (name, kind, currency, opening_balance, scope) "
                "VALUES (%s,%s,%s,%s,%s)",
                (a["name"], a["kind"], a["currency"], a["opening_balance"], a["scope"]))
        counts["accounts"] = len(data["accounts"])
        acc_id = {n: i for i, n in conn.execute("SELECT id, name FROM accounts").fetchall()}

        for r in data["recurring"]:
            conn.execute(
                "INSERT INTO recurring (title, amount, currency, day_of_month, category_id, scope, note) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s)",
                (r["title"], r["amount"], r["currency"], r["day_of_month"],
                 cat_id[r["category"]], r["scope"], r["note"]))
        counts["recurring"] = len(data["recurring"])

        for g in data["goals"]:
            conn.execute(
                "INSERT INTO goals (title, target_amount, currency, due_date, saved_amount, scope) "
                "VALUES (%s,%s,%s,%s,%s,%s)",
                (g["title"], g["target_amount"], g["currency"], g["due_date"],
                 g["saved_amount"], g["scope"]))
        counts["goals"] = len(data["goals"])

        for b in data["budget_limits"]:
            conn.execute(
                "INSERT INTO budget_limits (period, amount, currency, scope) VALUES (%s,%s,%s,%s)",
                (b["period"], b["amount"], b["currency"], b["scope"]))
        counts["budget_limits"] = len(data["budget_limits"])

        for m in data["merchant_aliases"]:
            conn.execute(
                "INSERT INTO merchant_aliases (alias, category_id) VALUES (%s,%s)",
                (m["alias"], cat_id[m["category"]]))
        counts["merchant_aliases"] = len(data["merchant_aliases"])

        for f in data["family_rules"]:
            conn.execute(
                "INSERT INTO family_rules (key, value_num, currency, unit, document_key, scope) "
                "VALUES (%s,%s,%s,%s,%s,%s)",
                (f["key"], f["value_num"], f.get("currency"), f.get("unit"),
                 f["document_key"], f["scope"]))
        counts["family_rules"] = len(data["family_rules"])

        for t in data["transactions"]:
            conn.execute(
                "INSERT INTO transactions "
                "(ts, amount, currency, account_id, category_id, merchant, comment, scope, person_id) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (t["ts"], t["amount"], t["currency"], acc_id[t["account"]],
                 cat_id[t["category"]], t.get("merchant"), t.get("comment"),
                 t["scope"], person_id.get(t.get("person"))))
        counts["transactions"] = len(data["transactions"])

    return counts

# ===== 5. ИНСТРУМЕНТЫ БЮДЖЕТА =====
#
# Семь функций вида f(ctx, ...) -> {"data": ..., "source_keys": [...]}. Каждая
# обязана подмешивать фильтр видимости в сам SQL-запрос (scope = ANY(%s) со
# списком из visible_scopes), а не отфильтровывать результат после выборки —
# приватность обеспечивается на уровне запроса, иначе личные данные супруга
# успевают попасть в память процесса до фильтрации.
#
# Баланс счёта нигде не хранится отдельным полем: opening_balance плюс сумма
# транзакций по счёту, вычисляется в каждом запросе заново — иначе хранимое
# поле и фактическая сумма транзакций неизбежно разойдутся.

def _scopes(ctx: Ctx) -> list[str]:
    return list(visible_scopes(ctx.person, ctx.chat_type))

def get_balance(ctx: Ctx) -> dict:
    """Остатки по счетам, раздельно по валютам. Баланс = opening_balance + сумма
    транзакций со знаком: расходы (categories.kind='expense') вычитаются,
    доходы прибавляются. Суммы в transactions.amount хранятся положительными
    во всех случаях — знак несёт categories.kind, а не сам amount, поэтому
    подзапрос обязан присоединять categories и применять CASE, а не просто
    суммировать amount как есть (иначе расходы прибавлялись бы к балансу
    вместо вычитания — было исправлено этой правкой)."""
    with db() as conn:
        cur = conn.execute(
            """SELECT a.currency,
                      SUM(a.opening_balance) + COALESCE(SUM(t.total), 0) AS balance
               FROM accounts a
               LEFT JOIN (SELECT t.account_id,
                                 SUM(CASE WHEN c.kind = 'expense' THEN -t.amount
                                          ELSE t.amount END) AS total
                          FROM transactions t JOIN categories c ON c.id = t.category_id
                          WHERE t.scope = ANY(%s) GROUP BY t.account_id) t ON t.account_id = a.id
               WHERE a.scope = ANY(%s) GROUP BY a.currency""",
            (_scopes(ctx), _scopes(ctx)))
        data = {cur_code: float(bal) for cur_code, bal in cur.fetchall()}
    return {"data": data, "source_keys": ["family_data"]}

def _month_bounds(year: int, month: int) -> tuple[_dt.date, _dt.date]:
    """Возвращает [начало, конец) календарного месяца."""
    start = _dt.date(year, month, 1)
    end = _dt.date(year + 1, 1, 1) if month == 12 else _dt.date(year, month + 1, 1)
    return start, end

def _add_months(year: int, month: int, delta: int) -> tuple[int, int]:
    idx = (year * 12 + (month - 1)) + delta
    return idx // 12, idx % 12 + 1

def _period_range(period: str) -> tuple[_dt.date, _dt.date]:
    """yesterday | this_month | last_month | YYYY-MM -> [начало, конец)."""
    today = _dt.date.today()
    if period == "yesterday":
        d = today - _dt.timedelta(days=1)
        return d, d + _dt.timedelta(days=1)
    if period == "this_month":
        return _month_bounds(today.year, today.month)
    if period == "last_month":
        y, m = _add_months(today.year, today.month, -1)
        return _month_bounds(y, m)
    m = re.match(r"^(\d{4})-(\d{2})$", period)
    if not m:
        raise ValueError(f"неизвестный период: {period!r}")
    return _month_bounds(int(m.group(1)), int(m.group(2)))

def get_expenses(ctx: Ctx, period: str, category: str | None = None) -> dict:
    """Расходы (kind='expense') в рублях за период, с разбивкой по категориям.
    Только RUB: инструмент работает в рублях, конвертация валют — отдельный
    инструмент курса ЦБ (задача 8), здесь её сознательно нет."""
    start, end = _period_range(period)
    params: list = [_scopes(ctx), start, end]
    cat_filter = ""
    if category:
        cat_filter = " AND c.name = %s"
        params.append(category)
    with db() as conn:
        cur = conn.execute(
            f"""SELECT c.name, SUM(t.amount)
                FROM transactions t JOIN categories c ON c.id = t.category_id
                WHERE c.kind = 'expense' AND t.currency = 'RUB' AND t.scope = ANY(%s)
                  AND t.ts >= %s AND t.ts < %s{cat_filter}
                GROUP BY c.name ORDER BY c.name""",
            params)
        rows = cur.fetchall()
    by_category = [(name, float(total)) for name, total in rows]
    total = float(sum(v for _, v in by_category))
    return {"data": {"period": period, "category": category, "total": total,
                     "by_category": by_category},
            "source_keys": ["family_data"]}

def get_recurring(ctx: Ctx) -> dict:
    """Активные регулярные платежи, видимые в текущей области."""
    with db() as conn:
        cur = conn.execute(
            """SELECT title, amount, day_of_month, note FROM recurring
               WHERE active AND scope = ANY(%s) ORDER BY day_of_month""",
            (_scopes(ctx),))
        rows = cur.fetchall()
    items = [{"title": t, "amount": float(a), "day_of_month": d, "note": n}
             for t, a, d, n in rows]
    monthly_total = float(sum(i["amount"] for i in items))
    return {"data": {"items": items, "monthly_total": monthly_total},
            "source_keys": ["family_data"]}

def get_goals(ctx: Ctx) -> dict:
    """Финансовые цели, видимые в текущей области."""
    with db() as conn:
        cur = conn.execute(
            """SELECT title, target_amount, saved_amount, due_date, currency FROM goals
               WHERE scope = ANY(%s) ORDER BY due_date NULLS LAST""",
            (_scopes(ctx),))
        rows = cur.fetchall()
    items = [{"title": t, "target_amount": float(target), "saved_amount": float(saved),
              "due_date": due.isoformat() if due else None, "currency": cur_code}
             for t, target, saved, due, cur_code in rows]
    return {"data": {"items": items}, "source_keys": ["family_data"]}

def get_family_rule(ctx: Ctx, key: str) -> dict:
    """Числовое или текстовое семейное правило, с ключом объясняющей заметки."""
    with db() as conn:
        row = conn.execute(
            "SELECT value_num, value_text, currency, unit, document_key FROM family_rules "
            "WHERE key = %s AND scope = ANY(%s)", (key, _scopes(ctx))).fetchone()
    if not row:
        return {"data": None, "source_keys": []}
    num, txt, curr, unit, doc = row
    return {"data": {"key": key, "value": float(num) if num is not None else txt,
                     "currency": curr, "unit": unit},
            "source_keys": [doc] if doc else []}

def get_budget_status(ctx: Ctx) -> dict:
    """Лимит месяца (period='monthly'), потрачено в рублях с начала текущего
    месяца, остаток. Только RUB — конвертация валют не входит в этот
    инструмент, см. get_expenses."""
    today = _dt.date.today()
    start, end = _month_bounds(today.year, today.month)
    scopes = _scopes(ctx)
    with db() as conn:
        limit_row = conn.execute(
            "SELECT COALESCE(SUM(amount), 0) FROM budget_limits "
            "WHERE period = 'monthly' AND currency = 'RUB' AND scope = ANY(%s)", (scopes,)).fetchone()
        limit = float(limit_row[0])
        spent_row = conn.execute(
            """SELECT COALESCE(SUM(t.amount), 0) FROM transactions t
               JOIN categories c ON c.id = t.category_id
               WHERE c.kind = 'expense' AND t.currency = 'RUB' AND t.scope = ANY(%s)
                 AND t.ts >= %s AND t.ts < %s""",
            (scopes, start, end)).fetchone()
        spent = float(spent_row[0])
    remaining = limit - spent
    percent = (spent / limit * 100) if limit else 0.0
    last_day = end - _dt.timedelta(days=1)
    days_left = max(0, (last_day - today).days)
    return {"data": {"limit": limit, "spent": spent, "remaining": remaining,
                     "percent": percent, "days_left": days_left},
            "source_keys": ["family_data"]}

def forecast_cashflow(ctx: Ctx, months: int) -> dict:
    """Помесячная проекция рублёвого баланса: текущий баланс плюс средний
    доход минус средние расходы за последние три полных календарных месяца.
    gap_month — первый месяц, где прогнозный остаток на конец месяца уходит
    в минус, иначе None.

    Расходы — это просто среднее по фактическим расходным транзакциям за
    три месяца, без отдельного слагаемого для регулярных платежей:
    коммуналка и связь в сиде проведены обычными транзакциями и уже входят
    в это среднее, поэтому прежнее "recurring_total + avg_expenses" считало
    их дважды."""
    scopes = _scopes(ctx)
    today = _dt.date.today()

    balance = get_balance(ctx)["data"].get("RUB", 0.0)

    # Средние доход/расходы за последние три полных календарных месяца.
    # Только RUB — то же ограничение, что и у get_expenses/
    # get_budget_status: без него доллары молча складывались бы с рублями.
    y0, m0 = _add_months(today.year, today.month, -3)
    hist_start, _ = _month_bounds(y0, m0)
    hist_end, _ = _month_bounds(*_add_months(today.year, today.month, 0))
    with db() as conn:
        exp_row = conn.execute(
            """SELECT COALESCE(SUM(t.amount), 0) FROM transactions t
               JOIN categories c ON c.id = t.category_id
               WHERE c.kind = 'expense' AND t.currency = 'RUB' AND t.scope = ANY(%s)
                 AND t.ts >= %s AND t.ts < %s""",
            (scopes, hist_start, hist_end)).fetchone()
        inc_row = conn.execute(
            """SELECT COALESCE(SUM(t.amount), 0) FROM transactions t
               JOIN categories c ON c.id = t.category_id
               WHERE c.kind = 'income' AND t.currency = 'RUB' AND t.scope = ANY(%s)
                 AND t.ts >= %s AND t.ts < %s""",
            (scopes, hist_start, hist_end)).fetchone()
    avg_expenses = float(exp_row[0]) / 3
    avg_income = float(inc_row[0]) / 3

    month_list = []
    gap_month = None
    year, month = today.year, today.month
    for _ in range(months):
        year, month = _add_months(year, month, 1)
        income = avg_income
        expenses = avg_expenses
        balance = balance + income - expenses
        label = f"{year:04d}-{month:02d}"
        month_list.append({"month": label, "income": income, "expenses": expenses,
                            "balance_end": balance})
        if gap_month is None and balance < 0:
            gap_month = label

    return {"data": {"months": month_list, "gap_month": gap_month},
            "source_keys": ["family_data"]}

# --- Каскад категоризации трат (три ступени, порядок обязателен) ---
#
# 1. Точное совпадение в merchant_aliases — ноль обращений к модели.
# 2. Мимо алиаса — классификация моделью по списку категорий.
# 3. Модель не уверена — агент переспрашивает, а ответ пользователя
#    дописывается в алиасы через add_expense(..., learn_alias=True): в
#    следующий раз тот же мерчант находится на первой ступени.

class CategoryGuess(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: str
    confident: bool

def categorize(ctx: Ctx, merchant_text: str) -> dict:
    """Каскад: алиас (0 токенов) → LLM → переспросить. Порядок обязателен.
    Личный алиас перекрывает общий: ORDER BY (m.scope = 'common') ставит
    строку своей области раньше общей, если алиас записан под обоими сразу —
    так муж и жена могут учить один и тот же текст мерчанта под разными
    категориями, не затирая друг друга (первичный ключ теперь (alias, scope),
    а не просто alias)."""
    key = merchant_text.strip().lower()
    with db() as conn:
        row = conn.execute(
            "SELECT c.name FROM merchant_aliases m JOIN categories c ON c.id = m.category_id "
            "WHERE m.alias = %s AND m.scope = ANY(%s) "
            "ORDER BY (m.scope = 'common') ASC LIMIT 1", (key, _scopes(ctx))).fetchone()
        if row:
            return {"category": row[0], "via": "alias", "suggestions": None}
        names = [n for (n,) in conn.execute(
            "SELECT name FROM categories WHERE kind='expense' ORDER BY name").fetchall()]
    guess = structured_call(CategoryGuess, [{"role": "user", "content":
        f"Отнеси трату «{merchant_text}» к одной из категорий: {', '.join(names)}. "
        f"confident=false, если уверенности нет."}])
    if guess.confident and guess.category in names:
        return {"category": guess.category, "via": "llm", "suggestions": None}
    return {"category": None, "via": "ask", "suggestions": names}

def _pick_account(conn, scope: str, currency: str) -> int:
    """Счёт для проводки траты: строго та же область и та же валюта — общие
    деньги не могут показывать разный баланс в зависимости от того, кто
    спрашивает, поэтому подстановка личной траты на общий счёт запрещена.
    Если такого счёта нет вовсе (личная трата в валюте, для которой нет
    персонального счёта — в текущих сид-данных у супругов личных валютных
    счетов нет), это ошибка ввода: такую трату вносят как общую, а не молча
    привязывают к чужому счёту."""
    row = conn.execute(
        "SELECT id FROM accounts WHERE scope = %s AND currency = %s ORDER BY id LIMIT 1",
        (scope, currency)).fetchone()
    if not row:
        raise ValueError(
            f"нет счёта с областью {scope!r} и валютой {currency!r} — "
            "такую трату внести некуда (личную трату в этой валюте вносят как общую)")
    return row[0]

def add_expense(ctx: Ctx, amount: float, merchant: str, *, category: str,
                 currency: str = "RUB", comment: str | None = None,
                 scope: str = "common", learn_alias: bool = False) -> dict:
    """Записывает трату. scope — область самой траты, а не область видимости
    того, кто её вносит: личная трата должна прийти со scope='husband'/'wife'
    от вызывающей стороны, иначе она молча осядет в общем бюджете. При
    learn_alias=True мерчант дописывается в merchant_aliases под той же
    категорией и областью — третья ступень каскада категоризации. Запись
    транзакции и дозапись алиаса — одна явная транзакция базы: сбой между
    ними не должен оставлять трату записанной без соответствующего обучения."""
    if not math.isfinite(amount) or amount <= 0:
        raise ValueError("сумма расхода должна быть конечным положительным числом")
    with db() as conn:
        cat_row = conn.execute(
            "SELECT id FROM categories WHERE name = %s", (category,)).fetchone()
        if not cat_row:
            raise ValueError(f"неизвестная категория: {category!r}")
        category_id = cat_row[0]
        account_id = _pick_account(conn, scope, currency)
        person_id = resolve_person_id(conn, ctx.person)
        with conn.transaction():
            tx_row = conn.execute(
                """INSERT INTO transactions
                   (amount, currency, account_id, category_id, merchant, comment, scope, person_id)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                (amount, currency, account_id, category_id, merchant, comment, scope,
                 person_id)).fetchone()
            tx_id = tx_row[0]
            if learn_alias:
                conn.execute(
                    """INSERT INTO merchant_aliases (alias, category_id, scope) VALUES (%s,%s,%s)
                       ON CONFLICT (alias, scope) DO UPDATE SET
                         category_id = EXCLUDED.category_id""",
                    (merchant.strip().lower(), category_id, scope))
    return {"data": {"id": tx_id, "category": category, "amount": float(amount)},
            "source_keys": ["family_data"]}

PositiveMoney = Annotated[FiniteFloat, Field(gt=0)]

class _SpendingCheck(BaseModel):
    """Внутренняя схема для structured_call: в отличие от публичной
    ParsedSpending несёт ещё is_spending, чтобы модель могла сказать «это не
    про трату» — публичный parse_spending превращает такой ответ в None.
    Полей без умолчаний: если модель прислала is_spending=true, но не
    заполнила amount/merchant/currency, это ошибка разбора (уйдёт в повтор
    structured_call), а не молчаливая трата на 0 рублей без названия."""
    model_config = ConfigDict(extra="forbid")
    is_spending: bool
    amount: FiniteFloat
    merchant: str
    currency: Literal["RUB", "USD"]

    @model_validator(mode="after")
    def validate_spending_amount(self):
        if self.is_spending and self.amount <= 0:
            raise ValueError("сумма расхода должна быть положительной")
        return self

class ParsedSpending(BaseModel):
    model_config = ConfigDict(extra="forbid")
    amount: PositiveMoney
    merchant: str
    currency: Literal["RUB", "USD"]

def parse_spending(text: str) -> ParsedSpending | None:
    """Разбирает свободный текст о трате («потратил 3200 в ВкусВилле») в
    amount/merchant/currency. None, если сообщение не о трате (вопрос,
    команда, разговор) — тогда его дальше ведут по другому пути.

    Раунд ревью 2 задачи 12: наличие суммы в тексте — не признак траты.
    Классификатор раньше судил по этому одному сигналу и путал вопросы с
    числом («Отпуск обойдётся в 280 тысяч — тянем?») с отчётом о
    совершённом расходе — в 4 из 6 живых прогонов такая фраза уходила в
    учёт расходов вместо агента (см. demo_scenarios.md, сценарий 7).
    Признак траты — не число, а утверждение о том, что деньги УЖЕ
    потрачены; промпт ниже называет модели оба класса сигналов явно и в
    общем виде (по классам формулировок, а не по конкретным фразам)."""
    guess = structured_call(_SpendingCheck, [{"role": "user", "content":
        "Классифицируй сообщение: это отчёт об уже совершённой трате денег "
        "или что-то другое (вопрос, прикидка, план, рассуждение)?\n\n"
        "Признак траты — утверждение о том, что деньги УЖЕ потрачены: "
        "обычно прошедшее время, совершённый вид («потратил», «купил», "
        "«заплатил», «отдал», «отправил»), с указанием, на что или где "
        "потрачено. Само по себе число в тексте тратой не делает — вопрос "
        "с суммой тратой не является, сколько бы чисел он ни содержал.\n\n"
        "Признаки НЕ-траты, даже если в тексте есть сумма: вопросительный "
        "знак и вопросительные слова («хватит ли», «потянем», «тянем», "
        "«сколько», «стоит ли», «можно ли»); будущее или сослагательное "
        "время («обойдётся», «будет стоить», «планируем», «собираемся», "
        "«хотим»); прикидки, планы и рассуждения о будущих или "
        "гипотетических тратах. Это не отчёт о трате, даже если по форме "
        "похоже на одно предложение с суммой и местом.\n\n"
        "Если сообщение — отчёт о совершённой трате: выдели сумму, "
        "продавца и валюту (RUB или USD, по умолчанию RUB), is_spending=true. "
        "Если это не про совершённую трату — is_spending=false, а amount=0, "
        "merchant=\"\", currency=\"RUB\" (поля обязательны в любом случае, "
        "даже когда это не трата). "
        f"Сообщение: «{text}»"}])
    if not guess.is_spending:
        return None
    return ParsedSpending(amount=guess.amount, merchant=guess.merchant, currency=guess.currency)

# ===== 6. RAG =====
# --- Парсер корпусов ---

_DOC_HEAD = re.compile(r"^## ((?:PF|HH)-\d+)\.\s*(.+?)\s*$", re.M)
_FIELD = re.compile(r"^\*\*(.+?):\*\*\s*(.*)$", re.M)
_NEXT_HEADING = re.compile(r"^#{1,2}[ \t]", re.M)   # обрезает тело на любом чужом заголовке 1-2 уровня
_PREFIX = {"knowledge": "PF", "household": "HH"}

def parse_corpus(text: str, doc_type: str) -> list[dict]:
    prefix = _PREFIX[doc_type]
    parts = _DOC_HEAD.split(text)
    docs = []
    for i in range(1, len(parts), 3):
        key, title, body = parts[i], parts[i + 1], parts[i + 2]
        if not key.startswith(prefix):
            raise ValueError(f"{key}: ожидался префикс {prefix} для doc_type={doc_type}")
        # Между документами PF-XXX/HH-XXX (или после последнего) корпус может содержать
        # посторонний раздел с собственным заголовком (например «Рекомендации по загрузке
        # в RAG»). Он не относится к статье/заметке — тело обрезаем на первом же таком
        # заголовке, а не тянем текст до следующего документа или до конца файла.
        m = _NEXT_HEADING.search(body)
        if m:
            body = body[:m.start()]
        fields = {k.strip(): v.strip() for k, v in _FIELD.findall(body)}
        if "Ключевые слова" not in fields:
            raise ValueError(f"{key}: нет поля «Ключевые слова»")

        if doc_type == "household":
            scope = fields.get("Область")
            if scope is None:
                raise ValueError(f"{key}: household-заметка без поля «Область»")
            if scope not in SCOPES:
                raise ValueError(f"{key}: недопустимая область {scope!r}, ожидалось одно из {SCOPES}")
        else:
            if "Область" in fields:
                raise ValueError(f"{key}: у knowledge-статьи не должно быть поля «Область»")
            scope = "common"

        keywords = [w.strip() for w in fields["Ключевые слова"].split(",") if w.strip()]
        known = {"Ключевые слова", "Область"}
        metadata = {k: v for k, v in fields.items() if k not in known}
        docs.append({
            "document_key": key, "title": title, "scope": scope,
            "text": body.strip().rstrip("-").strip(),
            "keywords": keywords, "metadata": metadata,
            "source": f"{key}. {title}",
        })
    return docs

def load_documents(conn, path: str, doc_type: str) -> int:
    """Индексация одного корпуса. Эмбеддинги считаются по одному документу —
    на 62 документах это занимает около восьми минут (см. README), поэтому
    каждый документ печатается по мере обработки: молчащая на восемь минут
    команда выглядит как зависание, и прогресс — не украшение, а то, что
    отличает рабочий --init от подвисшего в глазах проверяющего."""
    docs = parse_corpus(open(path, encoding="utf-8").read(), doc_type)
    total = len(docs)
    for i, d in enumerate(docs, start=1):
        print(f"  [{i}/{total}] {doc_type}: {d['document_key']} — {d['title']}", flush=True)
        conn.execute(
            """INSERT INTO documents
               (document_key, doc_type, title, source, scope, text, keywords, metadata, embedding)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (document_key) DO UPDATE SET
                 text=EXCLUDED.text, keywords=EXCLUDED.keywords,
                 metadata=EXCLUDED.metadata, embedding=EXCLUDED.embedding""",
            (d["document_key"], doc_type, d["title"], d["source"], d["scope"],
             d["text"], d["keywords"], json.dumps(d["metadata"], ensure_ascii=False),
             embed(f"{d['title']}. {d['text']}")))
    return total


# --- Поиск с фильтром видимости ---
# Ctx и visible_scopes определены в разделе 4 (ХРАНИЛИЩЕ) — используются как
# инструментами бюджета (раздел 5), так и поиском ниже.

def _search(conn, query: str, doc_type: str, ctx: Ctx, top_k: int) -> list[dict]:
    vec = embed(query)
    cur = conn.execute(
        """SELECT document_key, title, text, 1 - (embedding <=> %s::vector) AS score
           FROM documents
           WHERE doc_type = %s AND scope = ANY(%s)
           ORDER BY embedding <=> %s::vector
           LIMIT %s""",
        (vec, doc_type, list(visible_scopes(ctx.person, ctx.chat_type)), vec, top_k))
    return [{"document_key": k, "title": t, "text": x, "score": float(s)}
            for k, t, x, s in cur.fetchall()]

def search_knowledge(query: str, ctx: Ctx, top_k: int = 3) -> list[dict]:
    """Только общие принципы PF. Для семейных договорённостей — search_household."""
    with db() as conn:
        return _search(conn, query, "knowledge", ctx, top_k)

def search_household(query: str, ctx: Ctx, top_k: int = 3) -> list[dict]:
    """Только договорённости и история семьи HH. Для общих принципов — search_knowledge."""
    with db() as conn:
        return _search(conn, query, "household", ctx, top_k)

# ===== 7. MCP: БАНК РОССИИ =====
#
# Клиент внешнего MCP-сервера курсов ЦБ (atomno-mcp-cbr-rates, транспорт
# stdio, команда запуска — SETTINGS.mcp_cbr_cmd). Структура повторяет
# a06_mcp_client.py из учебных примеров: subprocess по stdio, JSON-RPC
# построчно. Отличие — после initialize отправляется уведомление
# notifications/initialized: без него часть серверов (в том числе этот)
# не отдаёт tools/list.
#
# Сервис внешний и может быть недоступен (сеть, PyPI и т.д.). Ни один из
# инструментов ниже не имеет права уронить агента — при сбое возвращается
# {"data": None, "error": ...}, чтобы агент мог сообщить об этом и продолжить
# работу с тем, что есть.
#
# Курс ЦБ — не курс обмена в банке. Здесь достаточно сохранить date и source
# из ответа сервера в source_keys, чтобы агент мог датировать цифру; сама
# оговорка про расхождение с банковским курсом — задача 9.
#
# Раунд ревью 1 добавил три вещи к первой версии:
#   1. Формирование результата (source_keys и т.п.) переехало ВНУТРЬ try —
#      раньше успешный call_tool с неожиданной формой ответа (сервер сменил
#      схему, версия в SETTINGS.mcp_cbr_cmd не совпала с ожиданиями кода)
#      бросал KeyError мимо всей защиты. Версия пакета в команде запуска
#      пинуется именно потому, что API может измениться, — канал реальный.
#   2. Таймаут на ожидание ответа: без него зависший сервер (не упавший,
#      а именно повисший — например, оборвалась сеть посреди запроса к
#      cbr.ru) блокирует однопоточного бота навсегда. Обычный вызов и
#      инициализация (первый запуск через uvx ощутимо дольше) используют
#      разные значения, оба настраиваются через окружение.
#   3. Блокировка вокруг создания синглтона: сейчас в проекте нет ни потоков,
#      ни asyncio, поэтому гонки быть не может, но задача 11 добавляет
#      асинхронный цикл Telegram — тогда два параллельных первых вызова
#      могли бы поднять два процесса и перемешать протокол по одному stdin.
#      Дешевле поставить блокировку заранее.

class MCPClient:
    """Клиент одного MCP-сервера по stdio. Процесс поднимается в конструкторе
    и живёт до close() — запуск через uvx занимает несколько секунд, поэтому
    инстанс переиспользуется (см. _mcp() ниже), а не создаётся на каждый вызов."""

    def __init__(self, command: str | None = None,
                init_timeout: float | None = None, call_timeout: float | None = None):
        cmd = shlex.split(command or SETTINGS.mcp_cbr_cmd)
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True, bufsize=1)
        self._id = 0
        self._io_lock = threading.Lock()
        self._call_timeout = call_timeout if call_timeout is not None else SETTINGS.mcp_call_timeout_seconds
        init_timeout = init_timeout if init_timeout is not None else SETTINGS.mcp_init_timeout_seconds
        try:
            self._send("initialize", {"protocolVersion": "2024-11-05", "capabilities": {},
                                      "clientInfo": {"name": "budget-agent", "version": "1.0.0"}},
                       expect_reply=True, timeout=init_timeout)
            self._send("notifications/initialized", None, expect_reply=False)
        except Exception:
            # Половина инициализации не должна оставлять осиротевший процесс —
            # никто, кроме нас, на self.proc ссылку ещё не получил.
            self.close()
            raise

    def _send(self, method, params=None, expect_reply=True, timeout=None):
        with self._io_lock:
            req = {"jsonrpc": "2.0", "method": method}
            request_id = None
            if expect_reply:
                self._id += 1
                request_id = self._id
                req["id"] = request_id
            if params:
                req["params"] = params
            self.proc.stdin.write(json.dumps(req) + "\n")
            self.proc.stdin.flush()
            if not expect_reply:
                return None
            wait = timeout if timeout is not None else self._call_timeout
            ready, _, _ = select.select([self.proc.stdout], [], [], wait)
            if not ready:
                raise TimeoutError(f"MCP-сервер не ответил за {wait:g} сек ({method})")
            line = self.proc.stdout.readline()
            if not line:
                err = self.proc.stderr.read()
                raise RuntimeError(f"MCP-сервер не ответил: {err.strip()[:500]}")
            response = json.loads(line)
            if response.get("id") != request_id:
                raise RuntimeError(
                    f"MCP нарушил JSON-RPC: ожидался id={request_id}, получен {response.get('id')!r}")
            return response

    def list_tools(self) -> list[dict]:
        return self._send("tools/list", {})["result"]["tools"]

    def call_tool(self, name: str, arguments: dict) -> dict:
        resp = self._send("tools/call", {"name": name, "arguments": arguments})
        if "error" in resp:
            raise RuntimeError(resp["error"].get("message", "ошибка MCP"))
        content = resp["result"]["content"]
        if resp["result"].get("isError"):
            raise RuntimeError(content[0].get("text", "ошибка MCP")[:500])
        return json.loads(content[0]["text"])

    def close(self):
        try:
            self.proc.terminate(); self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()

_mcp_singleton: "MCPClient | None" = None
_mcp_lock = threading.Lock()

def _mcp() -> MCPClient:
    """Один процесс сервера на всё время работы: запуск через uvx занимает
    секунды. Двойная проверка под блокировкой — на случай параллельных первых
    обращений из будущего асинхронного цикла (задача 11); сейчас в проекте
    нет потоков/asyncio, но дешевле подстраховаться заранее, чем ловить два
    поднятых процесса и перемешанный протокол по одному stdin позже."""
    global _mcp_singleton
    if _mcp_singleton is None:
        with _mcp_lock:
            if _mcp_singleton is None:
                _mcp_singleton = MCPClient()
    return _mcp_singleton

def cbr_get_rate(ctx: Ctx, char_code: str, on_date: str | None = None) -> dict:
    """Курс ЦБ. Не курс обмена в банке — это оговаривается в ответе пользователю (задача 9)."""
    args = {"char_code": char_code}
    if on_date:
        args["on_date"] = on_date
    try:
        data = _mcp().call_tool("get_rate", args)
        return {"data": data, "source_keys": [f"cbr:{char_code}@{data['date']}"]}
    except Exception as e:
        return {"data": None, "source_keys": [], "error": f"курс получить не удалось: {e}"}

def cbr_key_rate(ctx: Ctx) -> dict:
    """Ключевая ставка ЦБ. Ответ сервера содержит date_from/date_to и точки
    ряда, а не единственную дату — источник датируется по date_to (последняя
    точка диапазона)."""
    try:
        data = _mcp().call_tool("key_rate", {})
        return {"data": data, "source_keys": [f"cbr:key_rate@{data['date_to']}"]}
    except Exception as e:
        return {"data": None, "source_keys": [], "error": f"ключевую ставку получить не удалось: {e}"}

def cbr_inflation(ctx: Ctx, year_from: int | None, year_to: int | None) -> dict:
    """Инфляция ЦБ за диапазон лет. Ответ датируется годами (year_from/year_to),
    у инфляции нет единственной даты публикации, как у курса или ставки."""
    args = {}
    if year_from is not None:
        args["year_from"] = year_from
    if year_to is not None:
        args["year_to"] = year_to
    try:
        data = _mcp().call_tool("inflation", args)
        return {"data": data, "source_keys": [f"cbr:inflation@{data['year_from']}-{data['year_to']}"]}
    except Exception as e:
        return {"data": None, "source_keys": [], "error": f"данные по инфляции получить не удалось: {e}"}

def mcp_tools_description() -> str:
    """Краткое описание инструментов ЦБ для системного промпта агента (задача 9).
    При недоступности сервера возвращает пояснение вместо падения — сюда
    приходит то же требование отказоустойчивости, что и к cbr_*: агент должен
    суметь сформировать системный промпт, даже если сервер сейчас не поднимается."""
    try:
        tools = _mcp().list_tools()
        return "\n".join(f"- {t['name']}: {t['description']}" for t in tools)
    except Exception as e:
        return f"Инструменты Банка России сейчас недоступны: {e}"

# ===== 8. SGR-ЦИКЛ =====
#
# Ход мысли задан схемой NextStep: модель заполняет поля по порядку —
# goal_progress и plan_remaining_steps идут раньше call, поэтому к моменту
# выбора инструмента модель уже сформулировала своими словами, чего ей не
# хватает. Порядок полей в классе — часть контракта, менять нельзя.
#
# Четыре отличия от учебного SGR-примера:
#   1. call — типизированное размеченное объединение (discriminator="tool"):
#      имя инструмента жёстко связано со схемой его аргументов, правильное
#      имя с выдуманными аргументами не пройдёт валидацию.
#   2. Разбор ответа честный, через structured_call, а не поиск JSON регэкспом.
#   3. Нет поля с числом оставшихся попыток — счётчиком владеет цикл
#      (range(max_steps)), модели незачем его выдумывать, и он не может
#      разойтись с тем, что реально осталось.
#   4. Финализация — отдельный вызов FinalAnswer: пользователю уходит связный
#      ответ, собранный по всей цепочке шагов, а не сырое рассуждение
#      последнего шага.

class GetBalanceCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: Literal["get_balance"]

class GetExpensesCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: Literal["get_expenses"]
    period: str
    category: str | None = None

class GetRecurringCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: Literal["get_recurring"]

class GetGoalsCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: Literal["get_goals"]

class GetFamilyRuleCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: Literal["get_family_rule"]
    # Замкнутый список ключей семейных правил (см. seed_data.json) — не
    # свободная строка: инструмент типизирован так же строго, как остальные,
    # выдуманный ключ модель прислать не сможет.
    key: Literal["large_purchase_threshold", "emergency_fund_target",
                "mandatory_monthly_expenses", "vacation_indexation_pct",
                "personal_money_share_pct", "overspend_threshold"]

class GetBudgetStatusCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: Literal["get_budget_status"]

class ForecastCashflowCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: Literal["forecast_cashflow"]
    months: int

class SearchKnowledgeCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: Literal["search_knowledge"]
    query: str

class SearchHouseholdCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: Literal["search_household"]
    query: str

class CbrGetRateCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: Literal["cbr_get_rate"]
    char_code: Literal["USD", "EUR", "CNY"]

class CbrInflationCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: Literal["cbr_inflation"]
    year_from: int | None = None
    year_to: int | None = None

class CbrKeyRateCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: Literal["cbr_key_rate"]

class NoToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: Literal["none"]

ToolCall = Annotated[Union[GetBalanceCall, GetExpensesCall, GetRecurringCall, GetGoalsCall,
                           GetFamilyRuleCall, GetBudgetStatusCall, ForecastCashflowCall,
                           SearchKnowledgeCall, SearchHouseholdCall, CbrGetRateCall,
                           CbrInflationCall, CbrKeyRateCall, NoToolCall],
                     Field(discriminator="tool")]


class NextStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    goal_progress: str
    plan_remaining_steps: Annotated[list[str], Field(min_length=1, max_length=5)]
    decision_summary: str
    call: ToolCall
    task_completed: bool


class FinalAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str
    details: list[str]
    scenarios: list[str]
    source_keys: list[str]


def _rag_result(hits: list[dict]) -> dict:
    """Приводит выдачу RAG к общей форме инструмента. Поиск выполняется один раз."""
    return {"data": [{"key": h["document_key"], "title": h["title"], "text": h["text"]}
                     for h in hits],
            "source_keys": [h["document_key"] for h in hits]}


TOOL_REGISTRY: dict[str, callable] = {
    "get_balance":       lambda ctx, a: get_balance(ctx),
    "get_expenses":      lambda ctx, a: get_expenses(ctx, a["period"], a.get("category")),
    "get_recurring":     lambda ctx, a: get_recurring(ctx),
    "get_goals":         lambda ctx, a: get_goals(ctx),
    "get_family_rule":   lambda ctx, a: get_family_rule(ctx, a["key"]),
    "get_budget_status": lambda ctx, a: get_budget_status(ctx),
    "forecast_cashflow": lambda ctx, a: forecast_cashflow(ctx, a["months"]),
    "search_knowledge":  lambda ctx, a: _rag_result(search_knowledge(a["query"], ctx)),
    "search_household":  lambda ctx, a: _rag_result(search_household(a["query"], ctx)),
    "cbr_get_rate":      lambda ctx, a: cbr_get_rate(ctx, a["char_code"]),
    "cbr_inflation":     lambda ctx, a: cbr_inflation(ctx, a.get("year_from"), a.get("year_to")),
    "cbr_key_rate":      lambda ctx, a: cbr_key_rate(ctx),
}

_TOOL_HELP = """\
- get_balance: остатки по счетам, без аргументов
- get_expenses(period, category=None): траты за период; period — 'yesterday' | 'this_month' | 'last_month' | 'YYYY-MM'
- get_recurring: активные регулярные платежи, без аргументов
- get_goals: финансовые цели, без аргументов
- get_family_rule(key): семейное правило по ключу — large_purchase_threshold, emergency_fund_target, mandatory_monthly_expenses, vacation_indexation_pct, personal_money_share_pct, overspend_threshold
- get_budget_status: лимит/потрачено/остаток за текущий календарный месяц, без аргументов
- forecast_cashflow(months): помесячный прогноз рублёвого баланса на months месяцев вперёд
- search_knowledge(query): поиск по общим принципам финансовой грамотности (документы PF)
- search_household(query): поиск по семейным договорённостям и истории (документы HH)
- cbr_get_rate(char_code): курс ЦБ на сегодня, char_code — USD | EUR | CNY
- cbr_inflation(year_from=None, year_to=None): инфляция ЦБ за диапазон лет
- cbr_key_rate: ключевая ставка ЦБ, без аргументов
- none: инструмент не нужен — использовать, когда данных для ответа уже достаточно"""


def _expense_category_names() -> list[str]:
    """Точные названия категорий расходов — модели нужна их фактическая
    форма (регистр, написание), иначе она подставит в get_expenses(category=...)
    правдоподобную, но не совпадающую строку и молча получит пустой результат
    вместо ошибки: c.name сравнивается точным равенством, без нормализации."""
    with db() as conn:
        return [n for (n,) in conn.execute(
            "SELECT name FROM categories WHERE kind = 'expense' ORDER BY name").fetchall()]


# Раунд ревью 1 задачи 12: пять сценариев демо-прогона (4, 7, 8, 9, 10)
# независимо провалились по одной и той же причине — плоский список
# инструментов в _TOOL_HELP называет ЧТО есть, но не говорит, КОГДА что
# уместно. Модель предпочитала конкретные семейные цифры общим PF-статьям
# (нет приоритета — конкретное выглядит убедительнее абстрактного) и не
# сверялась с ЦБ по крупным будущим тратам (никто не говорил, что это
# вообще стоит делать). Эвристики ниже закрывают это одной правкой
# промпта, в общем виде — они относятся к КЛАССАМ вопросов, а не к
# конкретным двенадцати сценариям, иначе это была бы подгонка под список,
# а не исправление прорехи.
_TOOL_HEURISTICS = """\
Эвристики выбора инструмента (общие правила, не список конкретных вопросов):
- Вопрос о принципе или о том, как правильно поступать в целом — свериcь \
с общим корпусом знаний (search_knowledge) В ДОПОЛНЕНИЕ к семейным данным, \
а не вместо них: конкретные цифры семьи подтверждают вывод, но не заменяют \
опору на общий принцип, если вопрос сформулирован как общий.
- Вопрос про крупную будущую трату или план (покупка, поездка, \
долгосрочная цель) — возьми курс валюты и/или инфляцию ЦБ как грубый \
внешний ориентир и явно назови его грубым ориентиром, а не точным \
прогнозом; курс и инфляция не заменяют семейные данные, а дополняют их.
- Вопрос о том, как принято именно в этой семье («у нас», «как мы \
считаем», конкретная история) — это семейный корпус (search_household), \
а не общий (search_knowledge).
- Если вызванный инструмент вернул пустой результат (ничего не найдено) — \
СНАЧАЛА проверь себя: период, категория и другие аргументы вызова названы \
именно так, как требуют инструменты (точные названия категорий, период в \
их формате), а не приблизительно? Пустой результат из-за неточного \
аргумента — это ошибка вызова, а не факт о видимости или о реальности: не \
приписывай его ни «не существует», ни «здесь не видно» — перевызови \
инструмент с точным аргументом. Только если аргументы были точными и \
результат всё равно пуст, работает следующее: для того, что в принципе \
может принадлежать конкретному человеку (личная цель, личный счёт, личная \
трата, чьи-то персональные накопления) пустой результат означает «здесь, \
в текущей области видимости, этого не видно», а НЕ «этого не существует» \
— область видимости урезана по чату/человеку, и такая сущность вполне \
может существовать и быть видна в другом чате. Не выдавай такой пустой \
результат за факт отсутствия («накоплений нет», «такой цели нет») — \
скажи прямо и точно, что именно не видно ОТСЮДА (например: «в этом чате \
данных по этой цели не видно — возможно, она заведена в личном чате»). \
Честное признание «здесь не видно» — это полноценный, завершённый ответ, \
а не провал."""


def who_gen(person: str | None) -> str:
    return {"husband": "мужа", "wife": "жены"}.get(person, "неизвестного пользователя")


def _visibility_note(ctx: Ctx) -> str:
    """Явно объясняет модели модель видимости по scope — без этого у неё нет
    оснований отличить «данных нет вообще» от «здесь не видно»: пустой
    результат инструмента она читает буквально, как факт отсутствия (item 4
    финальной волны правок). Ветка ниже — ровно по visible_scopes: личный
    чат мужа/жены даёт common+свой scope, всё остальное (групповой чат,
    незнакомый/пустой chat_type, чужой person) — только common, поэтому
    формулировка ветки-по-умолчанию не называет её «групповым чатом»
    буквально — сюда попадает и, например, приватный чат с person=None."""
    if ctx.chat_type == "private" and ctx.person in ("husband", "wife"):
        other = "жены" if ctx.person == "husband" else "мужа"
        return (
            f"Видимость данных в этом чате (личный чат {who_gen(ctx.person)}): общие "
            f"семейные данные (scope='common') плюс личные данные {who_gen(ctx.person)} "
            f"(scope='{ctx.person}'). Личные данные {other} (её/его отдельный счёт, "
            f"личные траты, личные цели) отсюда НЕ видны в принципе — они не пропали и не "
            f"отсутствуют, они просто заведены в другом личном чате."
        )
    return (
        "Видимость данных в этом чате: только общие семейные данные (scope='common'). "
        "Личные счета, личные траты и личные цели каждого из супругов "
        "(scope='husband'/'wife') отсюда НЕ видны в принципе — они не пропали и не "
        "отсутствуют, они просто заведены в личных чатах и не транслируются сюда."
    )


def _system_prompt(ctx: Ctx) -> str:
    who = who_gen(ctx.person)
    categories = ", ".join(_expense_category_names())
    today = _dt.date.today().isoformat()
    return (
        f"Ты — агент семейного бюджета. Отвечаешь от лица {who}, тип чата: {ctx.chat_type}. "
        f"Сегодняшняя дата: {today}. У тебя нет другого способа узнать текущую дату — не "
        "угадывай год по памяти при переводе относительных периодов («в июле», «в этом "
        "месяце», «за прошлый месяц») в аргументы инструментов: либо считай от даты выше, "
        "либо, где это возможно, используй готовые относительные периоды инструмента "
        "('this_month'/'last_month'/'yesterday') вместо самостоятельно собранной 'YYYY-MM'. "
        "Работаешь пошагово: на каждом шаге сначала честно оцени прогресс к цели и то, чего "
        "ещё не хватает, и только потом выбирай следующее действие. Никогда не выдумывай "
        "цифры и факты — используй только то, что реально вернули инструменты на предыдущих "
        "шагах.\n\n" + _visibility_note(ctx) +
        "\n\nДоступные инструменты:\n" + _TOOL_HELP +
        "\n\n" + _TOOL_HEURISTICS +
        f"\n\nТочные названия категорий расходов (для get_expenses.category, писать "
        f"строго как здесь, иначе фильтр ничего не найдёт): {categories}." +
        "\n\nИнструменты Банка России (через MCP):\n" + mcp_tools_description() +
        "\n\nКогда данных для ответа уже достаточно, поставь call.tool='none' и "
        "task_completed=true."
    )


def _history_text(history: list[dict]) -> str:
    if not history:
        return "(шагов ещё не было)"
    lines = []
    for i, h in enumerate(history, 1):
        if "error" in h:
            lines.append(f"{i}. сбой разбора ответа модели на этом шаге: {h['error']}")
        elif h.get("tool"):
            note = f" [инструмент отказал: {h['tool_error']}]" if h.get("tool_error") else ""
            lines.append(f"{i}. {h['step']} → {h['tool']}: "
                         f"{json.dumps(h['result'], ensure_ascii=False, default=str)}{note}")
        else:
            lines.append(f"{i}. {h['step']}")
    return "\n".join(lines)


def _step_prompt(question: str, history: list[dict]) -> str:
    return (f"Вопрос пользователя: «{question}»\n\nУже сделано на предыдущих шагах:\n"
            f"{_history_text(history)}\n\nОпиши прогресс, оставшийся план и следующий шаг.")


def _final_prompt(question: str, history: list[dict], available_sources: list[str]) -> str:
    # Модель возвращает только ключи — сами идентификаторы придумывает не она,
    # а инструменты по ходу шагов (см. registry в run_agent). Поэтому ей нужно
    # явно показать, что доступно, а не полагаться на то, что она угадает или
    # вспомнит текст вида 'family_data' по смыслу истории шагов.
    sources = ", ".join(available_sources) if available_sources else "(источников не было)"
    return (f"Вопрос пользователя: «{question}»\n\nСобранные по всем шагам данные:\n"
            f"{_history_text(history)}\n\nДоступные идентификаторы источников (ровно из "
            f"этого набора, ничего от себя): {sources}\n\n"
            "Сформулируй связный итоговый ответ по всей цепочке шагов, а не только по "
            "последнему. В source_keys перечисли только те идентификаторы из списка выше, "
            "которые реально подтверждают ответ — не выдумывай новые и не добавляй лишние.\n\n"
            "Пиши как для человека, а не как отчёт о работе инструментов: история шагов "
            "выше — это твоя внутренняя кухня, пользователю она не нужна и не видна нигде, "
            "кроме этого сообщения. Не упоминай в summary/details/scenarios названия "
            "инструментов и функций (get_balance, get_expenses, search_knowledge и т.п.), "
            "их параметры (period='2026-07', scope='husband' и т.п.) и сам факт «был "
            "выполнен вызов X» — вместо этого сразу говори о сути: какие цифры/факты "
            "нашлись и что они значат.\n\n"
            "scenarios заполняй, только если однозначного ответа нет и вопрос реально "
            "допускает несколько вариантов трактовки или решения (например, ответ зависит "
            "от того, что пользователь имел в виду, или от выбора между несколькими "
            "равноценными путями). Если ответ однозначен — оставь scenarios пустым "
            "списком: не выдумывай варианты ради того, чтобы поле не пустовало.")


def _is_empty_tool_result(data) -> bool:
    """Пустой результат инструмента в смысле "ничего не нашли" — не любой
    falsy: 0.0 в get_balance или num_days=0 — это реальный ответ, а не
    пустота. Пустота — это отсутствие строк/элементов там, где инструмент
    в принципе мог бы их вернуть: список (search_knowledge/search_household
    отдают data прямо списком) или контейнер с ключом items/by_category
    (get_goals/get_recurring/get_expenses)."""
    if data is None:
        return True
    if isinstance(data, list):
        return len(data) == 0
    if isinstance(data, dict):
        for key in ("items", "by_category"):
            if key in data:
                return len(data[key]) == 0
    return False


def run_agent(question: str, ctx: Ctx, on_step: "callable | None" = None,
              max_steps: int = 8) -> tuple[FinalAnswer, set[str], bool]:
    """SGR-цикл. Возвращает (FinalAnswer, registry, any_empty_result).
    registry копит идентификаторы источников, фактически вернувшиеся из
    инструментов по ходу шагов. Источники, которые FinalAnswer называет, но
    которых нет в registry, отбрасываются ниже: модель источники не
    сочиняет, она только выбирает из того, что реально было получено.

    any_empty_result — True, если хотя бы один вызванный инструмент вернул
    пустой результат (см. _is_empty_tool_result). Финальная волна правок,
    пункт «оговорка про урезанную видимость»: раньше решение «сказать, что
    здесь не видно» было отдано модели текстом эвристики в промпте — на
    живых прогонах модель следовала ей примерно в одном случае из пяти, а в
    остальных либо отрицала существование, либо (хуже) называла ложную
    цифру вроде «сумма равна 0 рублей». render_answer берёт этот флаг
    вместе с ctx и registry и решает детерминированно, добавлять ли
    предупреждение о видимости — так же, как раньше решение об оговорке
    про курс ЦБ было отдано коду (needs_disclaimer), а не тексту модели.

    Раунд ревью 1 добавил три вещи:
      1. Вызов TOOL_REGISTRY[name] обёрнут в try/except — любое исключение
         инструмента (сеть недоступна у embed() при search_*, неверный формат
         периода у get_expenses, что угодно ещё) превращается в тот же вид
         результата, что уже возвращают cbr_*-инструменты при сбое:
         {"data": None, "source_keys": [], "error": ...}. Цикл это переживает
         естественным образом — модель на следующем шаге видит пустой
         результат и пометку об отказе, а не падение агента.
      2. Финальный вызов structured_call(FinalAnswer, ...) тоже обёрнут:
         два неудачных разбора подряд на финализации не должны ронять то
         самое место, где пользователь ждёт ответ.
      3. final.source_keys подставляется из registry, если модель вернула
         пустой список, а реестр непуст (см. финальный блок) — но НЕ если
         модель вернула непустой, пусть и неполный список: полноту
         цитирования агент не форсирует, это осознанный предел архитектуры.

    Раунд ревью 2: реестр раньше ехал в приватном pydantic-атрибуте
    FinalAnswer._used_registry — незаметно для схемы модели, но и незаметно
    терялся при выходе за пределы процесса (model_validate_json после
    сериализации восстанавливает объект без приватных атрибутов, тихо
    откатываясь к пустому реестру). Теперь registry — явное второе значение
    возврата: вызывающая сторона обязана пронести его сама до render_answer,
    единственного места, где решается нужна ли оговорка (needs_disclaimer)
    по факту вызова инструмента, а не по тому, что модель включила в
    source_keys — иначе модель могла бы подавить обязательную оговорку,
    просто не назвав источник."""
    system = _system_prompt(ctx)
    history: list[dict] = []
    registry: set[str] = set()
    any_empty_result = False
    for i in range(max_steps):
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": _step_prompt(question, history)}]
        try:
            step = structured_call(NextStep, msgs)
        except StructuredFailure as e:
            history.append({"error": str(e)})
            continue                    # контролируемый отказ шага, не падение агента
        if on_step:
            on_step(step, i + 1)
        if step.task_completed:
            break
        name = step.call.tool
        if name == "none":
            history.append({"step": step.decision_summary, "result": None})
            continue
        args = step.call.model_dump(exclude={"tool"})
        try:
            result = TOOL_REGISTRY[name](ctx, args)
        except Exception as e:
            result = {"data": None, "source_keys": [], "error": f"инструмент {name} не сработал: {e}"}
        registry.update(result.get("source_keys", []))
        # Отказ инструмента (сеть, MCP, что угодно) — не то же самое, что
        # видимость: считаем "пусто" только когда инструмент реально
        # отработал и честно вернул ничего, а не когда он упал с ошибкой —
        # иначе временный сбой CBR давал бы ложное "здесь не видно".
        if not result.get("error") and _is_empty_tool_result(result["data"]):
            any_empty_result = True
        history.append({"step": step.decision_summary, "tool": name, "result": result["data"],
                        "tool_error": result.get("error")})

    try:
        final = structured_call(FinalAnswer, [
            {"role": "system", "content": system},
            {"role": "user", "content": _final_prompt(question, history, sorted(registry))}])
    except StructuredFailure:
        # Контролируемый отказ финализации — тот же принцип, что и на шаге:
        # честный ответ о том, что не получилось, с тем, что успели собрать,
        # а не необработанное исключение там, где пользователь ждёт ответ.
        final = FinalAnswer(
            summary="Не удалось сформулировать связный ответ — модель не справилась с "
                    "финализацией. Ниже то, что удалось собрать по ходу шагов.",
            details=[f"{h['tool']}: {json.dumps(h['result'], ensure_ascii=False, default=str)}"
                     for h in history if h.get("tool")],
            scenarios=[], source_keys=sorted(registry))

    # источники, которых нет в реестре шага, отбрасываются; если модель не
    # назвала ни одного (но реестр не пуст) — подставляется реестр целиком.
    # Неполный, но непустой список модели остаётся как есть: полноту
    # цитирования не форсируем.
    final.source_keys = [k for k in final.source_keys if k in registry]
    if not final.source_keys and registry:
        final.source_keys = sorted(registry)
    return final, registry, any_empty_result


# ---- Оговорка и человекочитаемые ссылки на источники ----
#
# Оговорку ставит код по детерминированному правилу, а не модель: решение
# «нужна ли оговорка» слишком важно, чтобы оставлять его на усмотрение LLM.

INVEST_KEYS = ("инвест", "акци", "облигац", "портфел", "доходност", "вклад под")

def needs_disclaimer(question: str, source_keys: list[str]) -> str | None:
    """Чистая, детерминированная функция без обращения к LLM — но её ответ на
    cbr-ветку верен ровно настолько, насколько верен переданный source_keys.
    Раунд ревью 1: вызывающая сторона (render_answer) обязана передавать
    сюда реестр фактически вызванных инструментов, а не то, что модель
    решила процитировать в FinalAnswer.source_keys, — иначе модель может
    подавить обязательную оговорку, просто не назвав cbr-источник (курс
    в тексте есть, ссылки нет — оговорки тоже нет). Инвестиционная ветка
    от этого не зависит: она смотрит только на текст вопроса."""
    q = question.lower()
    if any(k in q for k in INVEST_KEYS):
        return ("Это образовательное объяснение, а не персональная инвестиционная "
                "рекомендация.")
    if any(k.startswith("cbr:") for k in source_keys):
        return "Курс приведён по данным ЦБ; в банке курс обмена будет отличаться."
    return None


def _describe_source(key: str, titles: dict[str, str]) -> str:
    """Человекочитаемое описание одного идентификатора источника. Формат
    cbr:<предмет>@<датировка> неоднороден по датировке — у курса и ключевой
    ставки она одиночная дата, у инфляции диапазон лет, — поэтому разбор
    ветвится по предмету, а не ожидает единый формат хвоста после '@'."""
    if key == "family_data":
        return "данные семьи"
    if key.startswith("cbr:"):
        subject, _, when = key[len("cbr:"):].partition("@")
        if subject == "key_rate":
            return f"ключевая ставка ЦБ на {when}" if when else "ключевая ставка ЦБ"
        if subject == "inflation":
            return f"инфляция ЦБ за {when}" if when else "инфляция ЦБ"
        # иначе subject — код валюты (USD/EUR/CNY): это курс ЦБ на дату
        return f"курс ЦБ {subject} на {when}" if when else f"курс ЦБ {subject}"
    title = titles.get(key)
    return f"«{title}» ({key})" if title else key


def render_answer(ans: FinalAnswer, ctx: Ctx, question: str, registry: "set[str]",
                  any_empty_result: bool) -> str:
    """Собирает связный ответ пользователю: сводку, детали, сценарии,
    человекочитаемые ссылки на источники и — при необходимости — оговорку
    про курс ЦБ и/или про урезанную видимость.
    Заголовки документов достаёт сама по document_key из таблицы documents,
    ограничившись видимой ctx областью — второй, независимый барьер поверх
    того, что в source_keys и так не может быть ничего за пределами registry
    из run_agent.

    registry — обязательный параметр, без умолчания. Второе значение,
    которое вернул run_agent (см. его докстринг, раунд ревью 2).
    needs_disclaimer считается только по нему, а не по ans.source_keys: тот
    заполняет модель, и цитирование может быть неполным (модель не обязана
    называть каждый источник, полноту цитирования не форсируем — run_agent).
    Оговорка не имеет права зависеть от того, вспомнила ли модель сослаться
    на курс ЦБ.

    Раунд ревью 2 сделал registry именованным параметром с дефолтом None и
    откатом на ans.source_keys при пропуске — это убрало тихую потерю при
    сериализации, но оставило тихий путь возврата того же дефекта: пропуск
    аргумента на вызове ничем не сигналил и молча включал ровно ту логику
    (решение по source_keys модели), которую весь раунд 1 и закрывал.
    Раунд ревью 3: параметр обязателен, без значения по умолчанию — пропуск
    на вызове теперь TypeError прямо в месте ошибки, а не тихий откат к
    худшему поведению. Тесты, где FinalAnswer собран вручную без run_agent
    (инструменты не вызывались), передают явный set() — это не обход
    требования, а честное «в этом сценарии реестр пуст».

    any_empty_result — третье значение, которое вернул run_agent (см. его
    докстринг). Финальная волна правок: эвристика в системном промпте
    просила модель честно сказать «здесь не видно», если видимость урезана
    и данных нет, — на живых прогонах модель следовала этому примерно в
    одном случае из пяти, а в остальных либо отрицала существование
    личных данных, либо (хуже) называла ложную цифру вроде «сумма 0
    рублей». Оговорка про видимость ниже добавляется кодом безусловно по
    ctx+any_empty_result, а не по тому, упомянула ли модель это в тексте —
    тот же принцип, что и needs_disclaimer про курс ЦБ."""
    doc_keys = [k for k in ans.source_keys if k != "family_data" and not k.startswith("cbr:")]
    titles: dict[str, str] = {}
    if doc_keys:
        with db() as conn:
            rows = conn.execute(
                "SELECT document_key, title FROM documents "
                "WHERE document_key = ANY(%s) AND scope = ANY(%s)",
                (doc_keys, list(visible_scopes(ctx.person, ctx.chat_type)))).fetchall()
        titles = dict(rows)

    parts = [ans.summary]
    if ans.details:
        parts.append("\n".join(f"- {d}" for d in ans.details))
    if ans.scenarios:
        # Без заголовка два списка ("- ...", "- ...") шли подряд и ничем не
        # отличались на вид: факты и предположения читались как одно и то
        # же. Заголовок явно маркирует смену регистра — что ниже уже не
        # факт, а один из возможных вариантов.
        parts.append("Возможные варианты:\n" + "\n".join(f"- {s}" for s in ans.scenarios))
    if ans.source_keys:
        refs = "; ".join(_describe_source(k, titles) for k in ans.source_keys)
        parts.append(f"Источники: {refs}.")
    # Оговорка про урезанную видимость — детерминированная, из ctx и
    # any_empty_result, а не из текста модели (см. докстринг выше и
    # докстринг run_agent). Условие: человек опознан (иначе "видимость
    # урезана" бессмысленно — не для кого её объяснять) И видна только
    # общая область (личный чат видит common+свой scope — там нечего
    # прятать ОТ САМОГО СЕБЯ, оговорка нужна только там, где видимость
    # реально меньше полной семейной картины) И хотя бы один инструмент в
    # этом прогоне честно вернул пустой результат — если данные нашлись,
    # прятать нечего и оговорка была бы шумом.
    if (ctx.person in ("husband", "wife")
            and set(visible_scopes(ctx.person, ctx.chat_type)) == {"common"}
            and any_empty_result):
        parts.append("В этом чате видны только общие данные семьи; личные записи "
                     "каждого из супругов здесь не показываются.")
    disclaimer = needs_disclaimer(question, sorted(registry))
    if disclaimer:
        parts.append(disclaimer)
    return "\n\n".join(parts)

# ===== 9. БЫСТРЫЙ ПУТЬ =====
# Задача 7 закладывает состояние диалога (dialog_state) — оно нужно каскаду
# категоризации, чтобы третья ступень («агент переспрашивает») могла понять
# следующее сообщение пользователя как ответ на висящий вопрос, а не как
# новую трату. Задача 10 добавляет остальное содержимое раздела: команды и
# отчёты (handle_command, render_status, render_report, daily_report,
# send_scheduled_report) — регулярные запросы не стоит гнать через SGR-цикл,
# они обслуживаются чистым SQL и шаблоном, без единого обращения к модели.
# Это не только требование задания, но и страховка: если на защите модель
# окажется медленной или недоступной, статус и отчёт всё равно ответят
# мгновенно.

STATE_TTL_SECONDS = 3600

def get_state(chat_id: int, user_id: int | None = None) -> dict:
    """Состояние диалога по chat_id. Протухает через STATE_TTL_SECONDS: если
    updated_at старше часа, считаем, что пользователь ушёл, не ответив на
    висящий вопрос, и возвращаем базовое состояние, а не забытый pending —
    иначе случайное сообщение через день привяжется к вопросу недельной
    давности."""
    with db() as conn:
        row = conn.execute(
            "SELECT state, pending, EXTRACT(EPOCH FROM (now() - updated_at)) "
            "FROM dialog_state WHERE chat_id = %s AND user_id = %s",
            (chat_id, user_id or 0)).fetchone()
    if not row or row[2] > STATE_TTL_SECONDS:
        return {"state": "base", "pending": None}
    return {"state": row[0], "pending": row[1]}

def set_state(chat_id: int, state: str, pending: dict | None = None,
              _age_seconds: int = 0, user_id: int | None = None) -> None:
    """Записывает состояние диалога. _age_seconds сдвигает updated_at в
    прошлое — используется только тестами протухания, в обычной работе не
    передаётся (updated_at = now())."""
    with db() as conn:
        conn.execute(
            """INSERT INTO dialog_state (chat_id, user_id, state, pending, updated_at)
               VALUES (%s, %s, %s, %s, now() - make_interval(secs => %s))
               ON CONFLICT (chat_id, user_id) DO UPDATE SET state=EXCLUDED.state,
                 pending=EXCLUDED.pending, updated_at=EXCLUDED.updated_at""",
            (chat_id, user_id or 0, state,
             json.dumps(pending, ensure_ascii=False) if pending else None,
             _age_seconds))


def render_status(ctx: Ctx) -> str:
    """Лимит месяца, потрачено, остаток — чистый SQL и форматирование, без
    единого обращения к модели: команды остаются мгновенными и не зависят
    от доступности LLM."""
    d = get_budget_status(ctx)["data"]
    return (f"Потрачено {d['spent']:,.0f} ₽ из {d['limit']:,.0f} ₽ ({d['percent']:.0f}%).\n"
            f"Остаток {d['remaining']:,.0f} ₽ на {d['days_left']} дн.").replace(",", " ")

def render_report(ctx: Ctx) -> tuple[str, dict]:
    """Расходы за текущий месяц по категориям плюс кнопка «Совет от ИИ» —
    сама кнопка ведёт в тяжёлый SGR-путь, здесь только SQL и текст."""
    rows = get_expenses(ctx, period="this_month", category=None)["data"]["by_category"]
    markup = {"inline_keyboard": [[{"text": "Совет от ИИ", "callback_data": "ai_advice"}]]}
    if not rows:
        # Пустой this_month (например, в первые дни месяца) раньше давал
        # заголовок без единой строки под ним — читалось как оборванный
        # вывод, а не как «трат пока нет».
        return "В этом месяце трат ещё не было.", markup
    lines = [f"{name}: {total:,.0f} ₽".replace(",", " ") for name, total in rows]
    return "Расходы за месяц:\n" + "\n".join(lines), markup

HELP_TEXT = (
    "/status — сколько потрачено из лимита\n"
    "/report — расходы по категориям и совет от ИИ\n"
    "/start — задать месячный лимит заново\n\n"
    "Или просто напишите: «потратил 850 в Пятёрочке», "
    "«хватит ли до зарплаты», «сначала гасить кредит или копить подушку».")

def handle_command(cmd: str, ctx: Ctx, arg: str | None) -> "str | tuple[str, dict]":
    """Быстрый детерминированный путь: команды и отчёты обслуживаются чистым
    SQL и шаблоном, ни одна ветка не обращается к модели —
    test_commands_do_not_call_llm подменяет structured_call функцией,
    роняющей тест, если это условие нарушено."""
    if cmd == "/start":
        set_state(ctx.chat_id, "awaiting_limit", user_id=ctx.user_id)
        return "Какой месячный лимит трат ставим? Ответьте числом в рублях."
    if cmd == "/status":
        return render_status(ctx)
    if cmd == "/report":
        return render_report(ctx)          # кортеж (текст, разметка)
    if cmd == "/help":
        return HELP_TEXT
    return f"Не знаю команду {cmd}. {HELP_TEXT}"

def daily_report(ctx: Ctx) -> str:
    """Сводка вчерашних трат — уходит в общий чат по расписанию. Область
    видимости принудительно нормализуется в общую (person=None,
    chat_type='group') независимо от того, какой ctx передал вызывающий:
    личные траты не должны попасть в рассылку ни при каких условиях, даже
    если функцию по ошибке вызовут с личным ctx."""
    common_ctx = Ctx(person=None, chat_type="group", chat_id=ctx.chat_id)
    r = get_expenses(common_ctx, period="yesterday", category=None)["data"]
    if not r["by_category"]:
        return "Вчера трат не было."
    lines = [f"  {name}: {total:,.0f} ₽".replace(",", " ") for name, total in r["by_category"]]
    status = get_budget_status(common_ctx)["data"]
    return ("Траты за вчера:\n" + "\n".join(lines) +
            f"\nВсего: {r['total']:,.0f} ₽".replace(",", " ") +
            f"\nС начала месяца {status['spent']:,.0f} из {status['limit']:,.0f} ₽ "
            f"({status['percent']:.0f}%).".replace(",", " "))

# Адрес Bot API Telegram — точка публичного протокола, одна и та же для
# всех ботов и всех окружений, а не параметр развёртывания. Поэтому это
# именованная константа кода, а не переменная окружения: вынос в конфигурацию
# подсказывал бы, что значение настраиваемое, а оно таковым не является.
TELEGRAM_API_BASE = "https://api.telegram.org"

def send_scheduled_report() -> None:
    """Точка входа для cron, а не отдельный демон:
    `0 9 * * * cd /path && python budget_agent.py --report-cron`.
    Без токена печатает отчёт в stdout — так проверяющий убеждается в
    работе, не заводя расписание и не трогая Telegram."""
    ctx = Ctx(person=None, chat_type="group", chat_id=0)
    text = daily_report(ctx)
    if not SETTINGS.telegram_token:
        print(text)
        return
    chat_id = os.getenv("TELEGRAM_REPORT_CHAT_ID")
    if not chat_id:
        print("TELEGRAM_REPORT_CHAT_ID не задан, отчёт не отправлен:\n" + text)
        return
    response = httpx.post(f"{TELEGRAM_API_BASE}/bot{SETTINGS.telegram_token}/sendMessage",
                          json={"chat_id": chat_id, "text": text},
                          proxy=SETTINGS.telegram_proxy, timeout=30)
    response.raise_for_status()
    if not response.json().get("ok"):
        raise RuntimeError("Telegram Bot API не подтвердил отправку отчёта")


# ===== 10. TELEGRAM =====
#
# Бот повторяет a08_telegram_bot.py: поллинг, очередь трансляции шагов с
# воркером (гарантирует порядок сообщений при параллельных await — конкурентные
# send_message из нескольких корутин порядок бы не гарантировали), прокси из
# окружения, раздельные HTTPXRequest для Bot API и getUpdates (в v20+
# python-telegram-bot Bot и Updater используют разные бэкенды: без явного
# прокси на get_updates_request поллинг в закрытой сети зависает без ошибки,
# хотя бот выглядит "запущенным"). route_message сама решает, куда вести
# сообщение — обработчики здесь только резолвят Ctx, зовут route_message или
# run_agent и отправляют результат.

_tg_logger = logging.getLogger("budget_agent.telegram")

# Раунд ревью 1: сообщение пользователю при непредвиденном исключении не должно
# нести текст самого исключения — psycopg на сбое подключения кладёт в str(e)
# хост, порт и имя пользователя базы (ровно то, что прячется в окружение, а
# не в код). Общая формулировка в чат, полная трассировка — в лог процесса
# через _tg_logger.exception (см. _on_text/_on_advice).
_ON_TEXT_ERROR_MSG = "Не получилось обработать сообщение, попробуйте ещё раз."
_ON_ADVICE_ERROR_MSG = "Не получилось подготовить совет, попробуйте ещё раз."
_UNAUTHORIZED_MSG = "Этот бот доступен только настроенным пользователям и разрешённым чатам."

def resolve_ctx(update) -> Ctx:
    """Ctx из объекта Update. Работает и с реальным telegram.Update
    (Message.chat_id, Message.chat.type, Message.from_user.id — ровно эти
    атрибуты), и с обычными объектами той же формы, которые подставляют
    тесты без подключения к Telegram — поля достаются через обычные атрибуты,
    без завязки на конкретный класс. Для callback_query (кнопка «Совет от
    ИИ») источник тот же набор полей, но у самого сообщения, на которое
    ответили, и у update.callback_query.from_user, а не update.message.

    person резолвится по telegram_id из SETTINGS (husband_tg_id/wife_tg_id),
    а не по роли из базы — telegram_id единственная известная на входе
    величина, Ctx.person это уже роль.

    chat_type идёт как есть, без нормализации к двум значениям: Telegram
    отдаёт private/group/supergroup/channel вперемешку, а приводить их
    здесь к паре состояний значило бы завести второй список изменяемых
    значений рядом с защитой в visible_scopes(), которая уже закрывается
    по умолчанию (allow-list на "private") и не нуждается в полноте чужого
    списка, чтобы остаться закрытой."""
    if getattr(update, "message", None) is not None:
        msg = update.message
        user = msg.from_user
    else:
        cq = update.callback_query
        msg, user = cq.message, cq.from_user
    person = None
    if SETTINGS.husband_tg_id is not None and user.id == SETTINGS.husband_tg_id:
        person = "husband"
    elif SETTINGS.wife_tg_id is not None and user.id == SETTINGS.wife_tg_id:
        person = "wife"
    return Ctx(person=person, chat_type=msg.chat.type, chat_id=msg.chat_id, user_id=user.id)


def _is_authorized_ctx(ctx: Ctx) -> bool:
    """Telegram закрыт по умолчанию: известный супруг и разрешённый чат."""
    if ctx.person not in ("husband", "wife"):
        return False
    if ctx.chat_type == "private":
        return True
    return ctx.chat_id in SETTINGS.allowed_group_chat_ids


def _try_complete_limit(text: str, ctx: Ctx) -> str | None:
    """Пытается завершить awaiting_limit. None, если text не похож на ответ
    (ни одной цифры) — тогда route_message не удерживает вопрос про лимит
    силой, а трактует сообщение как новое намерение (раздел «Состояние
    диалога» design-doc: свободный вопрос в состоянии ожидания сбрасывает
    pending, человек мог передумать отвечать и спросить другое). Разбор
    суммы детерминированный, без обращения к модели — тот же принцип
    "быстрого пути", что и у команд."""
    digits = re.sub(r"[^\d]", "", text)
    if not digits:
        return None
    amount = float(digits)
    with db() as conn:
        conn.execute(
            """INSERT INTO budget_limits (period, amount, currency, scope)
               VALUES ('monthly', %s, 'RUB', 'common')
               ON CONFLICT (period, scope) DO UPDATE SET amount = EXCLUDED.amount""",
            (amount,))
    set_state(ctx.chat_id, "base", user_id=ctx.user_id)
    return f"Готово. Месячный лимит {amount:,.0f} ₽.".replace(",", " ")


def _try_complete_category(text: str, ctx: Ctx, pending: dict) -> str | None:
    """Пытается завершить awaiting_category — третья ступень каскада
    категоризации (см. раздел 5): add_expense(..., learn_alias=True)
    дописывает мерчанта в merchant_aliases, чтобы в следующий раз он
    находился по алиасу без единого обращения к модели.

    Сравнение с именем категории регистронезависимое (пользователь печатает
    название руками по подсказке бота — точный регистр не гарантирован).
    None, если совпадения нет вовсе — тогда route_message не удерживает
    вопрос силой (раунд ревью 1: реask с сохранением состояния читался бы на
    демонстрации как поломка, если пользователь вместо ответа задал другой
    вопрос), а трактует сообщение как новое намерение."""
    answer = text.strip()
    names = _expense_category_names()
    match = next((n for n in names if n.lower() == answer.lower()), None)
    if match is None:
        return None
    r = add_expense(ctx, pending["amount"], pending["merchant"], category=match,
                     currency=pending.get("currency", "RUB"), scope="common", learn_alias=True)
    set_state(ctx.chat_id, "base", user_id=ctx.user_id)
    return (f"Записал: {match} — {r['data']['amount']:,.0f} ₽ "
            f"в «{pending['merchant']}».").replace(",", " ")


def _handle_spending(parsed: ParsedSpending, ctx: Ctx) -> str:
    """Трата, разобранная parse_spending, идёт в каскад категоризации
    (categorize). scope='common': свободный текст о трате не несёт сигнала
    личное/общее, а бытовые траты (магазин, такси, кафе) в подавляющем
    большинстве общие. Личный scope не угадывается из текста — он
    используется там, где уже нужен явно (add_expense вызывается напрямую
    со scope='husband'/'wife', как в тестах раздела 5), а не выводится
    эвристикой над свободным сообщением."""
    guess = categorize(ctx, parsed.merchant)
    if guess["category"] is None:
        set_state(ctx.chat_id, "awaiting_category",
                  {"amount": parsed.amount, "merchant": parsed.merchant, "currency": parsed.currency},
                  user_id=ctx.user_id)
        options = ", ".join(guess["suggestions"])
        return f"Не разобрал категорию для «{parsed.merchant}». Какая категория? Варианты: {options}."
    r = add_expense(ctx, parsed.amount, parsed.merchant, category=guess["category"],
                     currency=parsed.currency, scope="common", learn_alias=(guess["via"] == "llm"))
    return (f"Записал: {guess['category']} — {r['data']['amount']:,.0f} {parsed.currency} "
            f"в «{parsed.merchant}».").replace(",", " ")


def _reset_notice(state: str, pending: dict | None) -> str:
    """Текст-приставка, когда сообщение в состоянии ожидания не опознано
    как ответ и висящий вопрос сброшен (см. route_message, раунд ревью 1).
    Для awaiting_category теряется конкретная незавершённая трата —
    сказать об этом явно, иначе пользователь не поймёт, что она пропала, и
    будет считать её записанной. Для awaiting_limit ничего предметного не
    теряется (лимит просто не задан), короткая нейтральная реплика."""
    if state == "awaiting_category" and pending:
        return (f"Незавершённая трата {pending['amount']:,.0f} ₽ в «{pending['merchant']}» "
                f"не сохранена — категория не распознана.\n\n").replace(",", " ")
    if state == "awaiting_limit":
        return "Хорошо, к лимиту вернёмся позже — можно задать его снова через /start.\n\n"
    return ""


def route_message(text: str, ctx: Ctx, on_step=None) -> "str | tuple[str, dict]":
    """Единая точка входа для Telegram- и CLI-слоя. Порядок ветвей
    обязателен и закреплён тестами по каждой границе отдельно
    (test_command_bypasses_dialog_state,
    test_awaiting_state_intercepts_before_spending_parse,
    test_awaiting_limit_intercepts_before_spending_parse):

      1. команда — по префиксу "/", не зависит от состояния диалога;
      2. ответ на висящий вопрос диалога (dialog_state) — если не отфильтровать
         здесь раньше проверки траты, "Продукты" в ответ на вопрос о категории
         уйдёт в parse_spending и разберётся (или нет) как отдельная,
         бессмысленная трата;
      3. сообщение о трате (parse_spending);
      4. содержательный вопрос — SGR-цикл (run_agent).

    Порядок ветвей не переставляется. Раунд ревью 1 уточнил, что происходит
    ВНУТРИ шага 2, а не сам порядок: design-doc (раздел «Состояние диалога»)
    требует, чтобы свободный вопрос в состоянии ожидания сбрасывал pending —
    человек мог передумать отвечать и спросить другое. _try_complete_limit /
    _try_complete_category возвращают None, если текст не похож на ответ (не
    число / не совпадает ни с одной категорией) — тогда состояние сбрасывается
    здесь же, наружу уходит уведомление о потере незавершённой траты (только
    для awaiting_category, где реально было что терять), а само сообщение
    идёт дальше по тем же двум ступеням (трата → SGR), как будто состояния
    ожидания и не было. Раньше повторный опрос удерживал состояние вечно —
    пользователь, задавший другой вопрос вместо ответа, не получал на него
    ответа вовсе, пока не закроет висящий вопрос или не истечёт STATE_TTL_SECONDS.

    on_step пробрасывается в run_agent как есть, по умолчанию None — как у
    самого run_agent. Это единственное расширение фиксированной сигнатуры
    (text, ctx) -> str | tuple[str, dict]: Telegram-слой передаёт сюда
    колбэк трансляции шагов (см. _run_with_broadcast), а CLI и все тесты,
    вызывающие route_message с двумя позиционными аргументами, поведения не
    меняют — совместимость с уже написанными вызовами не нарушена."""
    text = text.strip()
    if text.startswith("/"):
        cmd, _, arg = text.partition(" ")
        return handle_command(cmd, ctx, arg.strip() or None)

    st = get_state(ctx.chat_id, ctx.user_id)
    notice = ""
    if st["state"] == "awaiting_limit":
        done = _try_complete_limit(text, ctx)
        if done is not None:
            return done
        notice = _reset_notice(st["state"], st["pending"])
        set_state(ctx.chat_id, "base", user_id=ctx.user_id)
    elif st["state"] == "awaiting_category":
        done = _try_complete_category(text, ctx, st["pending"])
        if done is not None:
            return done
        notice = _reset_notice(st["state"], st["pending"])
        set_state(ctx.chat_id, "base", user_id=ctx.user_id)

    parsed = parse_spending(text)
    if parsed:
        return notice + _handle_spending(parsed, ctx)

    ans, registry, any_empty_result = run_agent(text, ctx, on_step=on_step)
    return notice + render_answer(ans, ctx, text, registry, any_empty_result)


def _split_for_telegram(text: str, max_len: int = 3600) -> list[str]:
    """Режет текст на части ≤ max_len (запас относительно лимита Telegram
    4096) по последнему переносу строки, чтобы не разрывать абзац
    посередине. Тот же приём, что в a08_telegram_bot.py."""
    text = (text or "").strip()
    if not text:
        return []
    parts = []
    while len(text) > max_len:
        cut = text.rfind("\n", 0, max_len)
        if cut <= 0:
            cut = max_len
        parts.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    parts.append(text)
    return parts

async def _reply(bot, chat_id: int, text: str, markup: dict | None = None) -> None:
    """Отправляет текст частями ≤ лимита Telegram; inline-клавиатуру (формат
    handle_command/render_report — {"inline_keyboard": [[{"text","callback_data"}]]})
    вешает только на последнюю часть, чтобы кнопка не размножилась."""
    from telegram import InlineKeyboardMarkup, InlineKeyboardButton
    parts = _split_for_telegram(text) or [text]
    kb = None
    if markup:
        kb = InlineKeyboardMarkup([[InlineKeyboardButton(b["text"], callback_data=b["callback_data"])
                                     for b in row] for row in markup["inline_keyboard"]])
    for i, part in enumerate(parts):
        await bot.send_message(chat_id=chat_id, text=part,
                               reply_markup=kb if i == len(parts) - 1 else None)

async def _broadcast_thinking(bot, chat_id: int, text: str) -> None:
    """Шаг агента моноширинным блоком (<pre><code>) — как в
    a08_telegram_bot.py: демонстрирует ход SGR-цикла, не путается с
    финальным ответом по оформлению."""
    import html
    from telegram.constants import ParseMode
    for part in _split_for_telegram(text):
        await bot.send_message(chat_id=chat_id, text=f"<pre><code>{html.escape(part)}</code></pre>",
                               parse_mode=ParseMode.HTML)

async def _run_with_broadcast(bot, ctx: Ctx, make_call):
    """Запускает синхронный вызов make_call(on_step) в отдельном потоке —
    чтобы не блокировать event loop поллинга — и, если SETTINGS.broadcast_steps
    включён, параллельно шлёт каждый шаг агента моноширинным блоком через
    очередь с одним воркером (порядок сообщений гарантирован воркером, не
    параллельными send_message). make_call принимает on_step (может быть
    None) и сама решает, передавать ли его дальше в run_agent — так один и
    тот же помощник обслуживает и свободный текст (через route_message), и
    кнопку «Совет от ИИ» (прямой вызов run_agent)."""
    import asyncio, contextlib
    loop = asyncio.get_running_loop()
    if not SETTINGS.broadcast_steps:
        return await loop.run_in_executor(None, lambda: make_call(None))

    queue: "asyncio.Queue[str]" = asyncio.Queue()

    def on_step(step, i):
        loop.call_soon_threadsafe(queue.put_nowait, f"Шаг {i}: {step.decision_summary}")

    async def worker():
        while True:
            item = await queue.get()
            try:
                await _broadcast_thinking(bot, ctx.chat_id, item)
            finally:
                queue.task_done()

    worker_task = asyncio.create_task(worker())
    try:
        result = await loop.run_in_executor(None, lambda: make_call(on_step))
    finally:
        await queue.join()
        worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task
    return result


async def _on_text(update, context) -> None:
    """Общий обработчик и для CommandHandler (четыре команды), и для
    MessageHandler свободного текста — route_message сама решает по тексту,
    куда вести сообщение, поэтому дублировать разбор здесь незачем.

    Сверх брифа: непредвиденное исключение (не StructuredFailure — та уже
    гасится внутри run_agent контролируемым отказом, см. его докстринг) не
    должно молча съедаться диспетчером python-telegram-bot, оставляя
    пользователя без единого ответа. Тот же принцип, что у TOOL_REGISTRY в
    run_agent и у структурированного разбора: честное сообщение об ошибке,
    а не падение там, где ждут ответ."""
    ctx = resolve_ctx(update)
    if not _is_authorized_ctx(ctx):
        await _reply(context.bot, ctx.chat_id, _UNAUTHORIZED_MSG)
        return
    text = update.message.text or ""
    try:
        result = await _run_with_broadcast(
            context.bot, ctx, lambda on_step: route_message(text, ctx, on_step))
    except Exception:
        _tg_logger.exception("необработанное исключение в _on_text (chat_id=%s)", ctx.chat_id)
        await _reply(context.bot, ctx.chat_id, _ON_TEXT_ERROR_MSG)
        return
    if isinstance(result, tuple):
        body, markup = result
        await _reply(context.bot, ctx.chat_id, body, markup)
    else:
        await _reply(context.bot, ctx.chat_id, result)


ADVICE_QUESTION = ("Дай совет по нашим тратам за этот месяц: что стоит урезать и какой принцип "
                   "ведения бюджета здесь применим.")

async def _on_advice(update, context) -> None:
    """Кнопка «Совет от ИИ» под /report — callback_query, тяжёлый SGR-путь
    (см. render_report/handle_command). answer() гасит "часики" на кнопке в
    клиенте Telegram сразу, не дожидаясь ответа модели. Тот же перехват
    непредвиденных исключений, что и в _on_text, и по той же причине."""
    await update.callback_query.answer()
    ctx = resolve_ctx(update)
    if not _is_authorized_ctx(ctx):
        await _reply(context.bot, ctx.chat_id, _UNAUTHORIZED_MSG)
        return
    def call(on_step):
        ans, registry, any_empty_result = run_agent(ADVICE_QUESTION, ctx, on_step=on_step)
        return render_answer(ans, ctx, ADVICE_QUESTION, registry, any_empty_result)
    try:
        result = await _run_with_broadcast(context.bot, ctx, call)
    except Exception:
        _tg_logger.exception("необработанное исключение в _on_advice (chat_id=%s)", ctx.chat_id)
        await _reply(context.bot, ctx.chat_id, _ON_ADVICE_ERROR_MSG)
        return
    await _reply(context.bot, ctx.chat_id, result)


def run_telegram() -> None:
    """Поллинг Telegram: CommandHandler на четыре команды (/start, /status,
    /report, /help), MessageHandler на свободный текст, CallbackQueryHandler
    на кнопку «Совет от ИИ». Без токена — понятное сообщение и возврат, а не
    падение: в закрытом контуре без реального бота эта ветка не нужна, но не
    должна мешать остальным точкам входа (--check-backend, CLI-вопрос)."""
    if not SETTINGS.telegram_token:
        print("Ошибка: TELEGRAM_BOT_TOKEN не задан — Telegram-слой не запускается.")
        return
    from telegram import Update
    from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters
    from telegram.request import HTTPXRequest

    request = HTTPXRequest(proxy=SETTINGS.telegram_proxy)
    updates_request = HTTPXRequest(proxy=SETTINGS.telegram_proxy, read_timeout=40,
                                   connect_timeout=20, write_timeout=20, pool_timeout=20)
    app = (Application.builder().token(SETTINGS.telegram_token)
          .request(request).get_updates_request(updates_request).build())
    for cmd in ("start", "status", "report", "help"):
        app.add_handler(CommandHandler(cmd, _on_text))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _on_text))
    app.add_handler(CallbackQueryHandler(_on_advice, pattern="^ai_advice$"))
    print("Telegram-бот запущен. Ctrl+C для остановки.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


# ===== 11. ТОЧКА ВХОДА =====

# ----- предполётная проверка (--doctor) -----
#
# Читает переменные окружения напрямую через os.getenv, а не через SETTINGS —
# сознательно: SETTINGS в режиме доктора равен None (см. комментарий у
# присвоения модуля выше), потому что load_settings() завершает процесс через
# sys.exit() ровно в той ситуации, которую доктор должен диагностировать —
# при отсутствующем/неполном .env. Каждая проверка укладывается в секунды и
# идёт до восьминутной индексации (--init), чтобы её причины — не скопирован
# .env, не отвечает Postgres, не загружена эмбеддинг-модель, база не пустая,
# не отвечает языковая модель — были видны сразу, а не после ожидания.

DOCTOR_OK = "OK"
DOCTOR_PROBLEM = "ПРОБЛЕМА"
DOCTOR_WARN = "ВНИМАНИЕ"
DOCTOR_SKIP = "ПРОПУЩЕНО"

def _doctor_check_python() -> tuple[str, str]:
    v = sys.version_info
    ver = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) >= (3, 11):
        return DOCTOR_OK, ver
    return DOCTOR_PROBLEM, f"{ver} — нужен Python 3.11 или новее"

def _doctor_check_deps() -> tuple[str, str]:
    modules = ("httpx", "psycopg", "dotenv", "openai", "pgvector", "pydantic", "telegram")
    missing = []
    for mod in modules:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        return DOCTOR_PROBLEM, (f"не установлены: {', '.join(missing)} — выполните "
                                 ".venv/bin/pip install -r requirements.txt")
    return DOCTOR_OK, "все установлены"

def _doctor_check_env_file() -> tuple[str, str]:
    from dotenv import find_dotenv
    # Без usecwd=True: тот же способ поиска, что и у load_dotenv() без
    # аргументов на строке 22 (по фрейму вызова, то есть от каталога самого
    # budget_agent.py) — при рассинхроне между этой проверкой и тем, что
    # load_dotenv() уже реально загрузил при импорте, доктор мог бы соврать.
    path = find_dotenv()
    if not path:
        return DOCTOR_PROBLEM, "не найден — скопируйте .env.example в .env (см. README, шаг 3) либо запустите ./setup.sh"
    return DOCTOR_OK, "найден"

def _doctor_check_postgres() -> tuple[tuple[str, str], "psycopg.Connection | None"]:
    dsn = os.getenv("PG_DSN", "").strip()
    if not dsn:
        return (DOCTOR_PROBLEM, "не задана PG_DSN — задайте строку подключения в .env"), None
    try:
        conn = psycopg.connect(dsn, autocommit=True, connect_timeout=5)
    except Exception as e:
        return (DOCTOR_PROBLEM, f"нет соединения — {e}"), None
    try:
        pg_version = conn.execute("SHOW server_version").fetchone()[0]
        row = conn.execute(
            "SELECT default_version FROM pg_available_extensions WHERE name = 'vector'").fetchone()
        vec_version = row[0] if row and row[0] else None
        if vec_version:
            return (DOCTOR_OK, f"PG {pg_version}, pgvector {vec_version}"), conn
        return (DOCTOR_PROBLEM, f"PG {pg_version}, расширение pgvector недоступно на сервере"), conn
    except Exception as e:
        conn.close()
        return (DOCTOR_PROBLEM, f"ошибка проверки версии — {e}"), None

def _doctor_check_db_empty(conn) -> tuple[str, str]:
    if conn is None:
        return DOCTOR_PROBLEM, "пропущено — нет соединения с Postgres"
    try:
        exists = conn.execute("SELECT to_regclass('public.transactions')").fetchone()[0]
        if exists is None:
            return DOCTOR_OK, "схема не создана, будет создана при --init"
        state_user_id = conn.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name='dialog_state' "
            "AND column_name='user_id'").fetchone()
        if state_user_id is None:
            return DOCTOR_PROBLEM, (
                "схема устарела: нет dialog_state.user_id — пересоздайте базу "
                "(docker compose down -v, затем up -d и --init)")
        count = conn.execute("SELECT count(*) FROM transactions").fetchone()[0]
        if count == 0:
            return DOCTOR_OK, "схема создана, транзакций 0"
        return DOCTOR_PROBLEM, f"{count} транзакций — используйте --init --reset либо чистую базу"
    except Exception as e:
        return DOCTOR_PROBLEM, f"ошибка проверки — {e}"

def _doctor_check_ollama() -> tuple[str, str]:
    url = (os.getenv("EMBED_URL") or os.getenv("OLLAMA_URL") or "").strip().rstrip("/")
    if not url:
        return DOCTOR_PROBLEM, "не задан EMBED_URL/OLLAMA_URL"
    try:
        r = httpx.get(f"{url}/api/tags", timeout=5)
        r.raise_for_status()
        return DOCTOR_OK, "отвечает"
    except Exception as e:
        return DOCTOR_PROBLEM, f"недоступна по {url} — {e}"

def _doctor_check_embed_model() -> tuple[str, str]:
    model = os.getenv("EMBED_MODEL", "").strip()
    url = os.getenv("EMBED_URL", "").strip().rstrip("/")
    if not model or not url:
        return DOCTOR_PROBLEM, "не заданы EMBED_MODEL/EMBED_URL"
    try:
        r = httpx.post(f"{url}/api/embed", json={"model": model, "input": "проверка окружения"}, timeout=30)
        r.raise_for_status()
        vec = r.json()["embeddings"][0]
        return DOCTOR_OK, f"{model}, {len(vec)} измерений"
    except Exception as e:
        return DOCTOR_PROBLEM, f"{model} — {e}"

def _doctor_check_llm(skip: bool) -> tuple[str, str]:
    if skip:
        return DOCTOR_SKIP, "проверка пропущена флагом --no-llm"
    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    try:
        if provider == "ollama":
            url = os.getenv("OLLAMA_URL", "").strip().rstrip("/")
            model = os.getenv("OLLAMA_MODEL", "").strip()
            if not (url and model):
                return DOCTOR_PROBLEM, "не заданы OLLAMA_URL/OLLAMA_MODEL"
            client = OpenAI(base_url=f"{url}/v1", api_key="ollama")
        else:
            url = os.getenv("ROUTER_API_URL", "").strip()
            key = (os.getenv("ROUTER_API_KEY") or os.getenv("AGENT_PLATFORM_API_KEY") or "").strip()
            model = os.getenv("ROUTER_MODEL", "").strip()
            if not (url and key and model):
                return DOCTOR_PROBLEM, "не заданы ROUTER_API_URL/ROUTER_API_KEY/ROUTER_MODEL"
            client = OpenAI(base_url=url, api_key=key)
        client.chat.completions.create(model=model, messages=[{"role": "user", "content": "ping"}], max_tokens=5)
        return DOCTOR_OK, "отвечает"
    except Exception as e:
        status = getattr(e, "status_code", None)
        if status == 429:
            return DOCTOR_PROBLEM, "429: исчерпана квота, повторите позже"
        if status is not None:
            return DOCTOR_PROBLEM, f"{status}: {e}"
        return DOCTOR_PROBLEM, str(e)

def _doctor_check_uvx() -> tuple[str, str]:
    cmd = os.getenv("MCP_CBR_CMD", "uvx atomno-mcp-cbr-rates@0.1.10").strip()
    binary = (shlex.split(cmd) or ["uvx"])[0]
    path = shutil.which(binary)
    if path:
        return DOCTOR_OK, f"найден ({path})"
    return DOCTOR_PROBLEM, f"{binary} не найден в PATH — установите uv (https://docs.astral.sh/uv/)"

def _doctor_check_telegram_token() -> tuple[str, str]:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        return DOCTOR_WARN, "не задан — бот не запустится, режим командной строки работает"
    return DOCTOR_OK, "задан"

def cmd_doctor(no_llm: bool = False) -> int:
    """Печатает список предполётных проверок и возвращает код возврата:
    0, если проблем нет, 1 — если хотя бы одна проверка в состоянии
    ПРОБЛЕМА. ВНИМАНИЕ на код возврата не влияет — это то, что не мешает
    работе, но стоит знать (сейчас единственный такой случай — отсутствующий
    токен Telegram, при нём CLI-режим по-прежнему доступен)."""
    rows: list[tuple[str, str, str]] = []
    rows.append(("Python 3.11+", *_doctor_check_python()))
    rows.append(("Зависимости", *_doctor_check_deps()))
    rows.append(("Файл .env", *_doctor_check_env_file()))

    (pg_status, pg_detail), conn = _doctor_check_postgres()
    rows.append(("Postgres", pg_status, pg_detail))
    rows.append(("База пустая", *_doctor_check_db_empty(conn)))
    if conn is not None:
        conn.close()

    rows.append(("Ollama", *_doctor_check_ollama()))
    rows.append(("Эмбеддинг-модель", *_doctor_check_embed_model()))
    rows.append(("Языковая модель", *_doctor_check_llm(no_llm)))
    rows.append(("uvx (для MCP ЦБ)", *_doctor_check_uvx()))
    rows.append(("Токен Telegram", *_doctor_check_telegram_token()))

    for label, status, detail in rows:
        print(f"{label:<32} {status:<9} {detail}")

    return 1 if any(status == DOCTOR_PROBLEM for _, status, _ in rows) else 0


def cmd_check_backend() -> None:
    print(f"провайдер: {SETTINGS.provider}  модель: {_model_name()}")
    mode = detect_mode()
    for candidate in ("strict", "json_object", "prompt"):
        ok = _probe(candidate)   # уже в кэше для всего, что проверил detect_mode()
        print(f"  {candidate:<12} {'ДА' if ok else 'нет'}")
    print(f"выбран режим: {mode}")


def main() -> None:
    """Единственная точка входа CLI. --init создаёт схему (CREATE TABLE IF
    NOT EXISTS — только на пустой базе, миграций нет) и индексирует оба
    корпуса плюс сид; --check-backend определяет режим структурированного
    вывода локальной/облачной модели; --telegram поднимает поллинг;
    --report-cron шлёт отчёт по cron (см. README); свободный текст без
    флагов уходит в route_message тем же путём, что и сообщение в Telegram —
    Ctx.chat_id=0 достаточно для CLI: диалоговое состояние (awaiting_*) в
    демо-прогоне CLI не используется, а persons.id резолвится заново внутри
    инструментов по Ctx.person, не по chat_id. --doctor — предполётная
    проверка окружения перед --init (см. cmd_doctor); работает даже без
    .env, потому что обрабатывается раньше load_settings() (см. присвоение
    SETTINGS в начале файла) и не проходит через остальные ветки ниже."""
    import argparse
    p = argparse.ArgumentParser(description="Агент семейного бюджета")
    p.add_argument("question", nargs="?", help="вопрос для CLI-режима")
    p.add_argument("--init", action="store_true", help="создать схему и загрузить данные")
    p.add_argument("--reset", action="store_true",
                    help="вместе с --init: стереть сид-таблицы (transactions/accounts/"
                         "goals и т.д.) и перезалить seed_data.json поверх непустой базы")
    p.add_argument("--check-backend", action="store_true")
    p.add_argument("--telegram", action="store_true")
    p.add_argument("--report-cron", action="store_true")
    p.add_argument("--doctor", action="store_true",
                    help="предполётная проверка окружения (Python, зависимости, .env, "
                         "Postgres/pgvector, пустота базы, Ollama, эмбеддинг- и языковая "
                         "модель, uvx, токен Telegram) — секунды, до восьминутной --init")
    p.add_argument("--no-llm", action="store_true",
                    help="вместе с --doctor: пропустить пробный вызов языковой модели "
                         "(не тратит квоту)")
    p.add_argument("--as", dest="person", choices=["husband", "wife"], default="husband")
    p.add_argument("--chat", choices=["private", "group"], default="private")
    a = p.parse_args()
    if a.doctor:
        sys.exit(cmd_doctor(no_llm=a.no_llm))
    if a.init:
        with db() as conn:
            create_schema(conn)
            print("документов:", load_documents(conn, "knowledge.md", "knowledge")
                                + load_documents(conn, "household.md", "household"))
            try:
                print("сид:", load_seed(conn, force=a.reset))
            except SeedNotEmpty as e:
                # Без --reset повторный --init на уже заполненной базе не
                # ошибка сборки — это защитный отказ (в таблицах могут быть
                # настоящие траты пользователя). Печатаем понятную фразу
                # вместо трассировки, которую легко принять за поломку.
                print(f"сид: пропущен — {e}\n"
                      "Данные уже загружены. Документы и схема обновлены как обычно.\n"
                      "Чтобы стереть текущие транзакции/счета/цели и перезалить "
                      "seed_data.json поверх них, запустите: "
                      "budget_agent.py --init --reset")
        return
    if a.check_backend:
        return cmd_check_backend()
    if a.telegram:
        return run_telegram()
    if a.report_cron:
        return send_scheduled_report()
    if a.question:
        ctx = Ctx(person=a.person, chat_type=a.chat, chat_id=0)
        result = route_message(a.question, ctx)
        # route_message может вернуть (текст, inline_keyboard) для команд с
        # кнопкой (например /report) — CLI не умеет рисовать кнопки, поэтому
        # печатает только текстовую часть, а не сырой кортеж.
        text = result[0] if isinstance(result, tuple) else result
        print(text)
        return
    p.print_help()


if __name__ == "__main__":
    main()
