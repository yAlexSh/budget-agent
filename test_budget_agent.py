import pytest
from pydantic import BaseModel, ConfigDict
from budget_agent import extract_json, StructuredFailure, structured_call, load_settings

def test_extract_plain_json():
    assert extract_json('{"a": 1}') == {"a": 1}

def test_extract_from_markdown_fence():
    raw = '```json\n{"tool": "get_balance", "args": {}}\n```'
    assert extract_json(raw) == {"tool": "get_balance", "args": {}}

def test_extract_with_surrounding_prose():
    raw = 'Вот результат:\n```json\n{"x": 2}\n```\nГотово.'
    assert extract_json(raw) == {"x": 2}

def test_empty_content_raises():
    with pytest.raises(StructuredFailure):
        extract_json("")

def test_no_json_raises():
    with pytest.raises(StructuredFailure):
        extract_json("Не могу ответить")

def test_no_type_coercion():
    # Строки остаются строками: приведение "45 000" к числу запрещено
    assert extract_json('{"amount": "45 000"}') == {"amount": "45 000"}


class Tiny(BaseModel):
    model_config = ConfigDict(extra="forbid")
    city: str
    hot: bool

def test_structured_call_retries_on_schema_mismatch(monkeypatch):
    calls = []
    def fake(messages, **kw):
        calls.append(messages)
        return '{"town": "Paris"}' if len(calls) == 1 else '{"city": "Paris", "hot": true}'
    monkeypatch.setattr("budget_agent._raw_completion", fake)
    result = structured_call(Tiny, [{"role": "user", "content": "погода"}], mode="prompt")
    assert result.city == "Paris" and result.hot is True
    assert len(calls) == 2
    assert "town" in calls[1][-1]["content"], "текст ошибки должен уйти в повтор"

def test_structured_call_fails_after_second_attempt(monkeypatch):
    calls = []
    def fake(messages, **kw):
        calls.append(messages)
        return '{"nope": 1}'
    monkeypatch.setattr("budget_agent._raw_completion", fake)
    with pytest.raises(StructuredFailure):
        structured_call(Tiny, [{"role": "user", "content": "x"}], mode="prompt")
    assert len(calls) == 2, "ровно один повтор: две попытки, не одна и не три"

def test_structured_call_fails_on_empty_content(monkeypatch):
    calls = []
    def fake(messages, **kw):
        calls.append(messages)
        return ""
    monkeypatch.setattr("budget_agent._raw_completion", fake)
    with pytest.raises(StructuredFailure):
        structured_call(Tiny, [{"role": "user", "content": "x"}], mode="prompt")
    assert len(calls) == 2, "ровно один повтор: две попытки, не одна и не три"


def test_load_settings_requires_ollama_model(monkeypatch):
    monkeypatch.setenv("PG_DSN", "postgresql://x/y")
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_URL", "http://127.0.0.1:11434")
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.setenv("EMBED_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("EMBED_MODEL", "qwen3-embedding:0.6b")
    monkeypatch.setenv("MCP_CBR_CMD", "uvx atomno-mcp-cbr-rates@0.1.10")
    with pytest.raises(SystemExit) as exc_info:
        load_settings()
    assert "OLLAMA_MODEL" in str(exc_info.value)


from budget_agent import embed, EMBED_DIM, db, create_schema

def test_embed_dimension():
    v = embed("Подушка безопасности — три месячных расхода.")
    assert len(v) == EMBED_DIM == 1024
    assert all(isinstance(x, float) for x in v[:10])

def test_embed_similar_texts_closer_than_unrelated():
    import math
    def cos(a, b):
        return sum(x*y for x, y in zip(a, b)) / (math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(y*y for y in b)))
    a = embed("сколько потратили на продукты")
    b = embed("расходы на еду за месяц")
    c = embed("ремонт двигателя автомобиля")
    assert cos(a, b) > cos(a, c)

def test_schema_creates_all_tables():
    with db() as conn:
        create_schema(conn)
        cur = conn.execute("SELECT tablename FROM pg_tables WHERE schemaname='public'")
        tables = {r[0] for r in cur.fetchall()}
    assert {"persons", "accounts", "categories", "transactions", "recurring",
            "goals", "merchant_aliases", "family_rules", "budget_limits",
            "dialog_state", "documents"} <= tables

def test_documents_vector_is_1024():
    with db() as conn:
        cur = conn.execute(
            "SELECT atttypmod FROM pg_attribute "
            "WHERE attrelid='documents'::regclass AND attname='embedding'")
        assert cur.fetchone()[0] == 1024


def test_embed_wrong_dimension_raises(monkeypatch):
    class FakeResp:
        def raise_for_status(self):
            pass
        def json(self):
            return {"embeddings": [[0.1, 0.2, 0.3]]}   # заведомо не 1024 измерения

    def fake_post(*args, **kwargs):
        return FakeResp()

    monkeypatch.setattr("budget_agent.httpx.post", fake_post)
    with pytest.raises(RuntimeError):
        embed("любой текст")


from budget_agent import parse_corpus

PF_SAMPLE = """# База
---
## PF-006. Что такое подушка

**Ключевые слова:** резерв, подушка
**Короткий ответ:** ликвидный запас.

Тело статьи.

**Вопрос:** Телефон — это экстренно?
**Ответ:** только если для работы.

---
"""

HH_SAMPLE = """# Семья
---
## HH-010. Подарок к годовщине

**Ключевые слова:** подарок, накопление
**Область:** husband
**Короткий ответ:** цель 150 000 рублей.

Тело заметки.

**Вопрос:** Показывать ли жене?
**Ответ:** нет.

---
"""

def test_parse_knowledge_defaults_to_common():
    docs = parse_corpus(PF_SAMPLE, "knowledge")
    assert len(docs) == 1
    d = docs[0]
    assert d["document_key"] == "PF-006"
    assert d["title"] == "Что такое подушка"
    assert d["scope"] == "common"
    assert d["keywords"] == ["резерв", "подушка"]
    assert "Тело статьи" in d["text"]

def test_parse_household_requires_scope():
    d = parse_corpus(HH_SAMPLE, "household")[0]
    assert d["document_key"] == "HH-010"
    assert d["scope"] == "husband"

def test_household_without_scope_raises():
    broken = HH_SAMPLE.replace("**Область:** husband\n", "")
    with pytest.raises(ValueError, match="HH-010"):
        parse_corpus(broken, "household")

def test_household_with_bad_scope_raises():
    broken = HH_SAMPLE.replace("husband", "муж")
    with pytest.raises(ValueError, match="область"):
        parse_corpus(broken, "household")

def test_unknown_fields_go_to_metadata():
    extra = HH_SAMPLE.replace("**Область:** husband",
                              "**Область:** husband\n**Тип:** стратегия семьи")
    d = parse_corpus(extra, "household")[0]
    assert d["metadata"]["Тип"] == "стратегия семьи"

def test_real_corpora_parse_clean():
    kn = parse_corpus(open("knowledge.md", encoding="utf-8").read(), "knowledge")
    hh = parse_corpus(open("household.md", encoding="utf-8").read(), "household")
    assert len(kn) == 50 and len(hh) == 12
    assert all(d["scope"] in ("common", "husband", "wife") for d in hh)
    assert {d["scope"] for d in hh} >= {"common", "husband", "wife"}

_TRAILING_SECTION_SAMPLE = """# База
---
## PF-020. Пример статьи с хвостовым разделом

**Ключевые слова:** пример, хвост
**Короткий ответ:** демонстрация обрезки тела документа.

Основной текст статьи, который должен остаться.

**Вопрос:** Это часть статьи?
**Ответ:** да.

---

# Рекомендации по загрузке в RAG

1. Этот текст не должен попасть в тело документа PF-020.
2. Он описывает правила индексации, а не содержание статьи.
"""

def test_trailing_non_doc_section_excluded_from_text():
    d = parse_corpus(_TRAILING_SECTION_SAMPLE, "knowledge")[0]
    assert d["document_key"] == "PF-020"
    assert "Основной текст статьи, который должен остаться" in d["text"]
    assert "Рекомендации по загрузке в RAG" not in d["text"]
    assert "не должен попасть в тело документа" not in d["text"]

def test_real_corpora_last_docs_exclude_trailing_section():
    kn = parse_corpus(open("knowledge.md", encoding="utf-8").read(), "knowledge")
    hh = parse_corpus(open("household.md", encoding="utf-8").read(), "household")
    pf_050 = next(d for d in kn if d["document_key"] == "PF-050")
    hh_012 = next(d for d in hh if d["document_key"] == "HH-012")
    assert "Не позволять модели подменять точные расчёты" not in pf_050["text"]
    assert "Делить документ по заголовкам" not in hh_012["text"]

def test_real_corpora_document_count_unaffected_by_trailing_fix():
    kn = parse_corpus(open("knowledge.md", encoding="utf-8").read(), "knowledge")
    hh = parse_corpus(open("household.md", encoding="utf-8").read(), "household")
    assert len(kn) == 50
    assert len(hh) == 12


from budget_agent import visible_scopes, search_knowledge, search_household, Ctx

def test_visible_scopes_private_chat():
    assert set(visible_scopes("wife", "private")) == {"common", "wife"}
    assert set(visible_scopes("husband", "private")) == {"common", "husband"}

def test_visible_scopes_group_chat_shows_only_common():
    # В общем чате не видно даже собственного личного — рядом второй супруг
    assert visible_scopes("wife", "group") == ("common",)
    assert visible_scopes("husband", "group") == ("common",)

def test_search_knowledge_returns_only_pf():
    ctx = Ctx(person="husband", chat_type="private", chat_id=1)
    hits = search_knowledge("сколько денег держать в резерве", ctx)
    assert hits and all(h["document_key"].startswith("PF-") for h in hits)

def test_search_household_returns_only_hh():
    ctx = Ctx(person="husband", chat_type="private", chat_id=1)
    hits = search_household("как мы считаем отпуск", ctx)
    assert hits and all(h["document_key"].startswith("HH-") for h in hits)

def test_search_household_hides_wife_note_from_husband():
    ctx = Ctx(person="husband", chat_type="private", chat_id=1)
    hits = search_household("личная цель курсы сертификация", ctx, top_k=10)
    assert "HH-011" not in {h["document_key"] for h in hits}

def test_search_household_hides_personal_in_group():
    ctx = Ctx(person="husband", chat_type="group", chat_id=-1)
    hits = search_household("подарок к годовщине", ctx, top_k=10)
    assert "HH-010" not in {h["document_key"] for h in hits}


@pytest.mark.parametrize("chat_type", ["supergroup", "channel", "Group", "", None, "totally_unknown_type"])
def test_visible_scopes_unknown_chat_type_defaults_closed(chat_type):
    # allow-list: личная область открывается ТОЛЬКО при chat_type == "private".
    # Любое другое значение — включая незнакомое, пустое и None — обязано закрываться,
    # а не открываться по умолчанию (deny-list на "group" пропускает supergroup и т.п.).
    assert visible_scopes("husband", chat_type) == ("common",)
    assert visible_scopes("wife", chat_type) == ("common",)

def test_search_household_hides_personal_in_supergroup():
    # Сквозная проверка: Telegram отдаёт "supergroup" для выросших групповых чатов,
    # а не "group". Личная заметка мужа не должна доехать до результатов поиска.
    ctx = Ctx(person="husband", chat_type="supergroup", chat_id=-1)
    hits = search_household("подарок к годовщине", ctx, top_k=10)
    assert "HH-010" not in {h["document_key"] for h in hits}


from budget_agent import load_seed, db, create_schema, SeedNotEmpty

@pytest.fixture
def seeded_conn():
    # Каждый сид-тест сам обеспечивает нужное состояние базы — не полагается
    # на то, что до него по алфавиту/порядку отработал другой тест с load_seed.
    # Раньше test_seed_has_usd_account и соседние читали состояние, оставшееся
    # от test_seed_covers_eight_months, и поодиночке падали с UndefinedTable
    # на свежей базе без схемы.
    with db() as conn:
        create_schema(conn)
        load_seed(conn, "seed_data.json", force=True)
        yield conn

def test_seed_covers_eight_months(seeded_conn):
    cur = seeded_conn.execute("SELECT min(ts)::date, max(ts)::date, count(*) FROM transactions")
    lo, hi, n = cur.fetchone()
    assert n >= 400, "восемь месяцев трат — это сотни транзакций"
    assert (hi - lo).days >= 230

def test_seed_has_usd_account(seeded_conn):
    cur = seeded_conn.execute("SELECT count(*) FROM accounts WHERE currency='USD'")
    assert cur.fetchone()[0] >= 1, "валютные накопления нужны для вызовов MCP"

def test_seed_has_personal_scopes(seeded_conn):
    cur = seeded_conn.execute("SELECT DISTINCT scope FROM transactions")
    assert {r[0] for r in cur.fetchall()} >= {"common", "husband", "wife"}

def test_seed_has_family_rules(seeded_conn):
    cur = seeded_conn.execute("SELECT key, value_num, document_key FROM family_rules")
    rules = {k: (v, d) for k, v, d in cur.fetchall()}
    assert rules["large_purchase_threshold"][0] == 45000
    assert rules["large_purchase_threshold"][1] == "HH-001"
    assert rules["emergency_fund_target"][0] == 550000
    assert rules["emergency_fund_target"][1] == "HH-002"


def test_load_seed_without_force_refuses_nonempty(seeded_conn):
    # seeded_conn уже загружен (force=True в фикстуре) — база не пуста.
    with pytest.raises(SeedNotEmpty):
        load_seed(seeded_conn, "seed_data.json", force=False)

def test_load_seed_without_force_preserves_real_data(seeded_conn):
    # Ревьюер воспроизвёл: вставил "настоящую" транзакцию поверх сид-данных,
    # вызвал load_seed — транзакция пропала без предупреждения. Проверяем,
    # что без force=True она остаётся на месте, а вызов явно падает.
    acc_id = seeded_conn.execute("SELECT id FROM accounts LIMIT 1").fetchone()[0]
    cat_id = seeded_conn.execute("SELECT id FROM categories LIMIT 1").fetchone()[0]
    seeded_conn.execute(
        "INSERT INTO transactions (amount, currency, account_id, category_id, scope) "
        "VALUES (999999, 'RUB', %s, %s, 'common')", (acc_id, cat_id))
    with pytest.raises(SeedNotEmpty):
        load_seed(seeded_conn, "seed_data.json", force=False)
    cur = seeded_conn.execute("SELECT count(*) FROM transactions WHERE amount = 999999")
    assert cur.fetchone()[0] == 1, "реальная транзакция должна остаться нетронутой без force=True"

def test_load_seed_with_force_reloads_cleanly(seeded_conn):
    before = seeded_conn.execute("SELECT count(*) FROM transactions").fetchone()[0]
    counts = load_seed(seeded_conn, "seed_data.json", force=True)
    after = seeded_conn.execute("SELECT count(*) FROM transactions").fetchone()[0]
    assert before == after == counts["transactions"], "перезаливка не должна дублировать транзакции"

def test_load_seed_never_touches_documents(seeded_conn):
    before = seeded_conn.execute("SELECT count(*) FROM documents").fetchone()[0]
    with pytest.raises(SeedNotEmpty):
        load_seed(seeded_conn, "seed_data.json", force=False)
    after_refused = seeded_conn.execute("SELECT count(*) FROM documents").fetchone()[0]
    load_seed(seeded_conn, "seed_data.json", force=True)
    after_forced = seeded_conn.execute("SELECT count(*) FROM documents").fetchone()[0]
    assert before == after_refused == after_forced, "documents не входит в seed_data.json, трогать нельзя"


def test_seed_transaction_amounts_within_declared_ranges():
    # Диапазоны сумм из брифа: продукты 1500-6000, транспорт 300-900,
    # рестораны 2000-7000. Раньше масштабирование итоговой корзины под целевую
    # месячную сумму выталкивало часть сумм за эти границы.
    import json as _json
    ranges = {"Продукты": (1500, 6000), "Транспорт": (300, 900), "Рестораны": (2000, 7000)}
    data = _json.load(open("seed_data.json", encoding="utf-8"))
    violations = [t for t in data["transactions"]
                  if t["category"] in ranges
                  and not (ranges[t["category"]][0] <= t["amount"] <= ranges[t["category"]][1])]
    assert violations == [], f"суммы вне диапазона категории: {violations[:5]}"


import json

from budget_agent import (get_balance, get_expenses, get_budget_status,
                          forecast_cashflow, get_family_rule)

PRIV_H = Ctx(person="husband", chat_type="private", chat_id=1)
PRIV_W = Ctx(person="wife", chat_type="private", chat_id=2)
GROUP = Ctx(person="husband", chat_type="group", chat_id=-1)

def test_balance_computed_from_transactions():
    # Заменено по указанию ревьюера: исходный вариант теста повторял SQL
    # реализации (JOIN + фильтр по scope) и потому проверял сам себя — ошибка,
    # присутствующая одновременно и в реализации, и в тесте, всё равно давала
    # зелёный результат. Здесь ожидаемое значение считается независимо от SQL:
    # прямо из seed_data.json обычным Python-суммированием.
    #
    # Финальная волна, пункт 1: transactions.amount хранится положительным
    # всегда — знак несёт categories.kind (expense/income), а не сам amount.
    # Ожидание пересчитано соответственно: расходы вычитаются, доходы
    # прибавляются (раньше тест, как и сама реализация, просто складывал
    # amount — на счёте с одними расходами это давало то же самое случайное
    # совпадение, что и баг в SQL).
    #
    # "Личная карта мужа" — единственный счёт, который отличает PRIV_H (common +
    # husband) от GROUP (только common): его нет в общем чате. Поэтому разница
    # между двумя вызовами get_balance должна равняться ровно балансу этого
    # счёта: opening_balance плюс подписанная сумма его транзакций.
    data = json.load(open("seed_data.json", encoding="utf-8"))
    cat_kind = {c["name"]: c["kind"] for c in data["categories"]}
    card_txns = [t for t in data["transactions"] if t.get("account") == "Личная карта мужа"]
    assert card_txns, "в сид-данных должны быть транзакции по личной карте мужа"
    assert all(t["scope"] == "husband" for t in card_txns), \
        "все транзакции личного счёта мужа должны иметь scope='husband'"

    def signed(t):
        return -t["amount"] if cat_kind[t["category"]] == "expense" else t["amount"]

    expected_card_balance = 40000 + sum(signed(t) for t in card_txns)

    priv_rub = get_balance(PRIV_H)["data"]["RUB"]
    group_rub = get_balance(GROUP)["data"]["RUB"]
    assert abs((priv_rub - group_rub) - expected_card_balance) < 0.01

def test_balance_in_group_excludes_personal_accounts():
    # Переформулировано (финальная волна, пункт 1): прежний вариант
    # утверждал, что баланс в общем чате МЕНЬШЕ баланса в личном — это
    # держалось на дефекте (расходы прибавлялись к балансу, а не
    # вычитались). На знаковых суммах порядок величин может развернуться:
    # личные траты мужа (все — расходы) перекрывают остаток на его же
    # личной карте, и PRIV_H может оказаться меньше GROUP (см.
    # test_balance_computed_from_transactions — знак разницы больше не
    # гарантирован). Порядок величин — не то, что здесь проверяется по
    # сути; по сути проверяется РАЗНИЦА ВИДИМОСТИ: личный счёт мужа виден
    # только ему самому в личном чате, общий чат его не видит вовсе.
    assert set(visible_scopes(GROUP.person, GROUP.chat_type)) == {"common"}
    assert set(visible_scopes(PRIV_H.person, PRIV_H.chat_type)) == {"common", "husband"}

    # Повторное ревью: голое "!=" ниже проходит даже при откате get_balance
    # на беззнаковое SUM(amount) — GROUP и PRIV_H тогда тоже отличались бы
    # (просто на другую, тоже неверную величину), и регресс правки баланса
    # тест бы не заметил. Добавлена проверка КОНКРЕТНЫХ ожидаемых значений,
    # посчитанных независимо от SQL прямо по seed_data.json тем же
    # подписанным способом, что и в test_balance_computed_from_transactions/
    # test_balance_group_context_matches_full_seed_computation — если
    # реализация вернётся к беззнаковому суммированию, эти конкретные числа
    # разойдутся с фактом (расходы на личной карте мужа входили бы со
    # знаком "+" вместо "-"), и тест покраснеет.
    data = json.load(open("seed_data.json", encoding="utf-8"))
    cat_kind = {c["name"]: c["kind"] for c in data["categories"]}
    acc_opening = {a["name"]: a["opening_balance"] for a in data["accounts"]
                   if a["currency"] == "RUB"}

    def signed(t):
        return -t["amount"] if cat_kind[t["category"]] == "expense" else t["amount"]

    def expected_rub(scopes: set[str]) -> float:
        accounts = {a["name"] for a in data["accounts"]
                    if a["scope"] in scopes and a["currency"] == "RUB"}
        opening = sum(acc_opening[name] for name in accounts)
        txn_sum = sum(signed(t) for t in data["transactions"]
                      if t["scope"] in scopes and t["currency"] == "RUB")
        return opening + txn_sum

    expected_group = expected_rub({"common"})
    expected_priv_h = expected_rub({"common", "husband"})
    assert expected_group != expected_priv_h,         "предположение теста: личная карта мужа реально меняет баланс"

    assert abs(get_balance(GROUP)["data"]["RUB"] - expected_group) < 0.01
    assert abs(get_balance(PRIV_H)["data"]["RUB"] - expected_priv_h) < 0.01
    assert get_balance(GROUP)["data"]["RUB"] != get_balance(PRIV_H)["data"]["RUB"]

def test_expenses_by_category():
    r = get_expenses(PRIV_H, period="last_month", category="Продукты")
    assert r["data"]["total"] > 0
    assert r["data"]["category"] == "Продукты"

def test_budget_status_percent():
    r = get_budget_status(PRIV_H)["data"]
    assert r["limit"] == 140000
    assert 0 <= r["percent"] <= 300
    assert r["spent"] + r["remaining"] == pytest.approx(r["limit"], abs=1)

def test_family_rule_returns_number_and_source():
    r = get_family_rule(PRIV_H, "large_purchase_threshold")
    assert r["data"]["value"] == 45000
    assert r["source_keys"] == ["HH-001"]

def test_forecast_detects_gap_or_not():
    r = forecast_cashflow(PRIV_H, months=3)["data"]
    assert len(r["months"]) == 3
    assert all("balance_end" in m for m in r["months"])
    assert isinstance(r["gap_month"], (str, type(None)))


def test_forecast_does_not_double_count_recurring():
    # Финальная волна, пункт 2: avg_expenses (среднее по ВСЕМ расходным
    # транзакциям за три месяца) уже включает те же коммуналку и связь, что
    # и recurring_total из get_recurring — оба смотрят в одни и те же
    # исторические транзакции. Раньше expenses = recurring_total +
    # avg_expenses считал их дважды; починка — просто avg_expenses.
    # Ожидание считается независимо от forecast_cashflow: те же три
    # исторических месяца, прямая сумма из базы.
    import datetime as _dt
    scopes = list(visible_scopes(PRIV_H.person, PRIV_H.chat_type))
    today = _dt.date.today()
    recurring_total = get_recurring(PRIV_H)["data"]["monthly_total"]
    from budget_agent import _add_months, _month_bounds
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
    avg_expenses = float(exp_row[0]) / 3
    expected_month_expenses = avg_expenses

    r = forecast_cashflow(PRIV_H, months=1)["data"]
    actual_month_expenses = r["months"][0]["expenses"]
    assert abs(actual_month_expenses - expected_month_expenses) < 0.01
    # Регрессионная проверка сути дефекта: старая формула (recurring_total +
    # avg_expenses) не должна совпадать с новым результатом, если регулярные
    # платежи реально присутствуют среди исторических трат (recurring_total
    # > 0) — иначе тест не отличил бы починку от дефекта.
    assert recurring_total > 0
    buggy_month_expenses = recurring_total + avg_expenses
    assert abs(actual_month_expenses - buggy_month_expenses) > 0.01


# ===== Раунд ревью 1: находки 1-4 =====

from budget_agent import get_recurring, get_goals, _period_range


def test_balance_group_context_matches_full_seed_computation():
    # Находка 4: прежняя проверка баланса покрывала только РАЗНИЦУ между PRIV_H и
    # GROUP (вклад личного счёта мужа) — если бы общая (common) часть баланса
    # считалась неверно ОДИНАКОВО в обоих контекстах, разница осталась бы прежней
    # и тест бы этого не заметил. Здесь считаем ожидаемый общий баланс GROUP
    # целиком, независимо от SQL реализации — прямым Python-суммированием по
    # seed_data.json (в сид-данных scope транзакции всегда совпадает со scope её
    # счёта — проверено отдельно ниже, так что суммирование по account-множеству
    # и по scope-фильтру транзакций дают один и тот же набор).
    data = json.load(open("seed_data.json", encoding="utf-8"))
    acc_scope = {a["name"]: a["scope"] for a in data["accounts"]}
    assert all(acc_scope.get(t.get("account")) == t["scope"] for t in data["transactions"]), \
        "предположение теста: scope транзакции всегда совпадает со scope её счёта"

    # Финальная волна, пункт 1: та же поправка на знак, что и в
    # test_balance_computed_from_transactions — amount в базе всегда
    # положительный, знак несёт categories.kind.
    cat_kind = {c["name"]: c["kind"] for c in data["categories"]}
    common_rub_accounts = {a["name"] for a in data["accounts"]
                            if a["scope"] == "common" and a["currency"] == "RUB"}
    opening = sum(a["opening_balance"] for a in data["accounts"]
                  if a["name"] in common_rub_accounts)
    txn_sum = sum((-t["amount"] if cat_kind[t["category"]] == "expense" else t["amount"])
                  for t in data["transactions"]
                  if t["scope"] == "common" and t["currency"] == "RUB")
    expected_group_rub = opening + txn_sum

    actual_group_rub = get_balance(GROUP)["data"]["RUB"]
    assert abs(actual_group_rub - expected_group_rub) < 0.01


def test_expenses_total_is_float_when_empty():
    # Находка 3: sum() пустого списка даёт int 0, а форма из плана обещает float —
    # задачи 9-10 полагаются на это буквально.
    #
    # Финальная волна, пункт 3: раньше пустой выборкой служил "this_month" —
    # предпосылка держалась только на том, что сид обрывался в июле 2026 и
    # текущий месяц был пуст всегда. После починки пункта 3 (make_seed.py
    # теперь генерирует и неполный текущий месяц) это перестало быть
    # гарантией: в this_month есть настоящие траты, и "Образование" там
    # иногда тоже встречается — тест либо падает, либо проходит случайно, в
    # зависимости от дня генерации сида. Нужна выборка, пустая не по
    # везению, а по конструкции. "Зарплата" — единственная категория с
    # kind='income' (см. категории в seed_data.json) — get_expenses
    # фильтрует строго по c.kind='expense', поэтому по ней не найдётся ни
    # одной расходной транзакции ни в каком периоде, вне зависимости от
    # даты генерации сида. Период оставлен "this_month" осознанно: важна
    # пустота выборки по категории, а не конкретный период.
    r = get_expenses(PRIV_H, period="this_month", category="Зарплата")
    assert r["data"]["by_category"] == [], "категория дохода не должна иметь расходных строк"
    assert r["data"]["total"] == 0
    assert isinstance(r["data"]["total"], float)


def test_currency_filtered_tools_exclude_usd_transaction(seeded_conn):
    # Находка 1: get_expenses, get_budget_status (spent) и forecast_cashflow
    # (историческое среднее) суммировали t.amount без фильтра по валюте — рубли
    # молча складывались бы с долларами. Один инсёрт не может задеть все три
    # функции разом: у них разные временные окна (get_expenses/forecast смотрят
    # на "последний месяц"/"последние три месяца", get_budget_status — на текущий
    # месяц), поэтому вставляем по одной USD-транзакции в каждое окно.
    usd_acc = seeded_conn.execute(
        "SELECT id FROM accounts WHERE currency='USD' LIMIT 1").fetchone()[0]
    cat_id = seeded_conn.execute(
        "SELECT id FROM categories WHERE name='Продукты'").fetchone()[0]

    last_month_start, _ = _period_range("last_month")
    this_month_start, _ = _period_range("this_month")

    before_expenses = get_expenses(PRIV_H, period="last_month")["data"]["total"]
    before_status = get_budget_status(PRIV_H)["data"]["spent"]
    before_forecast = forecast_cashflow(PRIV_H, months=1)["data"]["months"][0]["expenses"]

    seeded_conn.execute(
        "INSERT INTO transactions (ts, amount, currency, account_id, category_id, scope) "
        "VALUES (%s, 999999, 'USD', %s, %s, 'common')", (last_month_start, usd_acc, cat_id))
    seeded_conn.execute(
        "INSERT INTO transactions (ts, amount, currency, account_id, category_id, scope) "
        "VALUES (%s, 888888, 'USD', %s, %s, 'common')", (this_month_start, usd_acc, cat_id))
    try:
        after_expenses = get_expenses(PRIV_H, period="last_month")["data"]["total"]
        after_status = get_budget_status(PRIV_H)["data"]["spent"]
        after_forecast = forecast_cashflow(PRIV_H, months=1)["data"]["months"][0]["expenses"]
        assert after_expenses == pytest.approx(before_expenses), \
            "USD-транзакция не должна попадать в get_expenses (рублёвый инструмент)"
        assert after_status == pytest.approx(before_status), \
            "USD-транзакция не должна попадать в get_budget_status.spent"
        assert after_forecast == pytest.approx(before_forecast), \
            "USD-транзакция не должна попадать в среднее forecast_cashflow"
    finally:
        seeded_conn.execute(
            "DELETE FROM transactions WHERE amount IN (999999, 888888) AND currency='USD'")


def test_goals_returns_expected_form():
    r = get_goals(PRIV_H)["data"]
    assert r["items"], "у мужа в личном чате должны быть видны хотя бы общие цели"
    item = r["items"][0]
    assert set(item) == {"title", "target_amount", "saved_amount", "due_date", "currency"}
    assert isinstance(item["target_amount"], float)
    assert isinstance(item["saved_amount"], float)


def test_goals_private_chat_sees_own_and_common():
    # Контрольный случай приватности задания: HH-010 (цель мужа "Подарок к
    # годовщине") и HH-011 (личная цель жены "Курсы и сертификация") не должны
    # смешиваться между личными чатами.
    titles_h = {g["title"] for g in get_goals(PRIV_H)["data"]["items"]}
    titles_w = {g["title"] for g in get_goals(PRIV_W)["data"]["items"]}
    assert titles_h == {"Отпуск 2027", "Подарок к годовщине"}
    assert titles_w == {"Отпуск 2027", "Курсы и сертификация"}
    assert "Курсы и сертификация" not in titles_h
    assert "Подарок к годовщине" not in titles_w


def test_goals_group_chat_sees_only_common():
    titles = {g["title"] for g in get_goals(GROUP)["data"]["items"]}
    assert titles == {"Отпуск 2027"}


def test_recurring_wife_scope_visible_only_to_wife_private():
    # "Абонемент в зал" — scope='wife': видим жене в личном чате, не видим мужу
    # (ни в личном, ни тем более в общем чате), не видим в общем чате вовсе.
    titles_w = {r["title"] for r in get_recurring(PRIV_W)["data"]["items"]}
    titles_h = {r["title"] for r in get_recurring(PRIV_H)["data"]["items"]}
    titles_group = {r["title"] for r in get_recurring(GROUP)["data"]["items"]}
    assert "Абонемент в зал" in titles_w
    assert "Абонемент в зал" not in titles_h
    assert "Абонемент в зал" not in titles_group


def test_recurring_monthly_total_is_float_when_empty(seeded_conn):
    # Тот же класс дефекта, что и в get_expenses (находка 3): sum() пустого
    # списка даёт int 0 вместо float. GROUP видит только recurring со
    # scope='common' — временно убираем их, чтобы получить пустой список, и
    # восстанавливаем в finally, чтобы не испортить данные для других тестов.
    rows = seeded_conn.execute(
        "SELECT title, amount, currency, day_of_month, category_id, scope, active, note "
        "FROM recurring WHERE scope = 'common'").fetchall()
    seeded_conn.execute("DELETE FROM recurring WHERE scope = 'common'")
    try:
        r = get_recurring(GROUP)["data"]
        assert r["items"] == []
        assert r["monthly_total"] == 0
        assert isinstance(r["monthly_total"], float)
    finally:
        for row in rows:
            seeded_conn.execute(
                "INSERT INTO recurring "
                "(title, amount, currency, day_of_month, category_id, scope, active, note) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s)", row)


def test_budget_status_limit_excludes_usd_budget_limit(seeded_conn):
    # Тот же класс дефекта, что и находка 1 (currency-фильтр у transactions):
    # budget_limits тоже допускает валюту помимо RUB. Лимит из get_budget_status
    # должен оставаться рублёвым инструментом — доллары не должны молча
    # складываться с рублями, когда появится личный лимит в USD.
    before = get_budget_status(PRIV_H)["data"]["limit"]
    seeded_conn.execute(
        "INSERT INTO budget_limits (period, amount, currency, scope) VALUES (%s,%s,%s,%s)",
        ("monthly", 999999, "USD", "husband"))
    try:
        after = get_budget_status(PRIV_H)["data"]["limit"]
        assert after == pytest.approx(before), \
            "USD-лимит не должен попадать в get_budget_status.limit (рублёвый инструмент)"
    finally:
        seeded_conn.execute(
            "DELETE FROM budget_limits WHERE period='monthly' AND scope='husband' AND currency='USD'")


# ===== Задача 7: каскад категоризации, ввод траты, состояние диалога =====

from budget_agent import (categorize, add_expense, parse_transaction,
                          get_state, set_state, STATE_TTL_SECONDS, resolve_person_id)

def test_alias_hit_costs_no_llm(monkeypatch):
    monkeypatch.setattr("budget_agent.structured_call",
                        lambda *a, **k: pytest.fail("модель не должна вызываться при попадании в алиас"))
    r = categorize(PRIV_H, "ВкусВилл")
    assert r["category"] == "Продукты" and r["via"] == "alias"

def test_unknown_merchant_goes_to_llm(monkeypatch):
    class R:
        category = "Продукты"
        confident = True
    monkeypatch.setattr("budget_agent.structured_call", lambda *a, **k: R())
    r = categorize(PRIV_H, "Ашан")
    assert r["category"] == "Продукты" and r["via"] == "llm"

def test_unconfident_llm_asks_user(monkeypatch):
    class R:
        category = "Продукты"
        confident = False
    monkeypatch.setattr("budget_agent.structured_call", lambda *a, **k: R())
    assert categorize(PRIV_H, "ООО Ромашка")["via"] == "ask"

def test_answering_category_saves_alias(monkeypatch):
    class R:
        category = "Продукты"
        confident = False
    monkeypatch.setattr("budget_agent.structured_call", lambda *a, **k: R())
    tx_id = None
    try:
        categorize(PRIV_H, "Новый магазин")
        r = add_expense(PRIV_H, 1200, "Новый магазин", category="Продукты", learn_alias=True)
        tx_id = r["data"]["id"]
        monkeypatch.setattr("budget_agent.structured_call",
                            lambda *a, **k: pytest.fail("после обучения модель не нужна"))
        assert categorize(PRIV_H, "Новый магазин")["via"] == "alias"
    finally:
        with db() as conn:
            if tx_id is not None:
                conn.execute("DELETE FROM transactions WHERE id=%s", (tx_id,))
            conn.execute("DELETE FROM merchant_aliases WHERE alias='новый магазин' AND scope='common'")

def test_state_roundtrip():
    set_state(42, "awaiting_category", {"amount": 3200, "merchant": "Ашан"})
    try:
        st = get_state(42)
        assert st["state"] == "awaiting_category"
        assert st["pending"]["merchant"] == "Ашан"
    finally:
        with db() as conn:
            conn.execute("DELETE FROM dialog_state WHERE chat_id=%s", (42,))

def test_stale_state_falls_back_to_base():
    set_state(43, "awaiting_category", {"amount": 1}, _age_seconds=STATE_TTL_SECONDS + 60)
    try:
        assert get_state(43)["state"] == "base"
    finally:
        with db() as conn:
            conn.execute("DELETE FROM dialog_state WHERE chat_id=%s", (43,))

def test_personal_expense_gets_person_scope():
    r = add_expense(PRIV_H, 2500, "Подарок", category="Подарки", scope="husband")
    try:
        with db() as conn:
            row = conn.execute("SELECT scope FROM transactions WHERE id=%s", (r["data"]["id"],)).fetchone()
        assert row[0] == "husband"
    finally:
        with db() as conn:
            conn.execute("DELETE FROM transactions WHERE id=%s", (r["data"]["id"],))


# ===== Раунд ревью 1: находки 1, 2, 3, 6 =====

def test_personal_usd_expense_without_account_raises_and_leaves_balance_intact():
    # У супругов нет личных долларовых счетов (находка 1 ревью) — привязка
    # личной валютной траты к общему счёту "Долларовая заначка" давала бы
    # разный баланс этого общего счёта в разных контекстах. Такую трату
    # вносить некуда — add_expense должен явно отказать, а не подставить
    # чужой счёт.
    with pytest.raises(ValueError, match="счёта"):
        add_expense(PRIV_H, 50, "Duty Free", category="Одежда", currency="USD", scope="husband")
    with db() as conn:
        n = conn.execute("SELECT count(*) FROM transactions WHERE merchant='Duty Free'").fetchone()[0]
    assert n == 0, "отказ должен случиться до записи — частичной траты быть не должно"
    b_h = get_balance(PRIV_H)["data"]["USD"]
    b_w = get_balance(PRIV_W)["data"]["USD"]
    b_g = get_balance(GROUP)["data"]["USD"]
    assert b_h == b_w == b_g, "баланс общего счёта не должен зависеть от того, кто спрашивает"

def test_spouses_learn_alias_independently_without_clobbering(monkeypatch):
    # Находка 2 ревью: муж и жена учат один и тот же текст мерчанта под
    # разными категориями — раньше общий PRIMARY KEY по alias затирал одну
    # запись поверх другой вместе с областью видимости.
    merchant = "Микс"
    tx_ids = []
    try:
        r1 = add_expense(PRIV_H, 500, merchant, category="Здоровье", scope="husband", learn_alias=True)
        tx_ids.append(r1["data"]["id"])
        r2 = add_expense(PRIV_W, 700, merchant, category="Развлечения", scope="wife", learn_alias=True)
        tx_ids.append(r2["data"]["id"])

        monkeypatch.setattr("budget_agent.structured_call",
                            lambda *a, **k: pytest.fail("алиас каждого супруга должен находиться без модели"))
        assert categorize(PRIV_H, merchant) == {"category": "Здоровье", "via": "alias", "suggestions": None}
        assert categorize(PRIV_W, merchant) == {"category": "Развлечения", "via": "alias", "suggestions": None}
    finally:
        with db() as conn:
            for tid in tx_ids:
                conn.execute("DELETE FROM transactions WHERE id=%s", (tid,))
            conn.execute("DELETE FROM merchant_aliases WHERE alias=%s", (merchant.lower(),))

def test_ctx_has_no_person_id_field():
    # Находка 3 ревью: person_id — мёртвое поле-ловушка, убрано из Ctx
    # совсем. Резолв по роли — отдельная функция resolve_person_id.
    assert not hasattr(PRIV_H, "person_id")
    with db() as conn:
        assert resolve_person_id(conn, "husband") is not None
        assert resolve_person_id(conn, None) is None

def test_categorize_suggestions_present_in_every_branch(monkeypatch):
    # Находка 6 ревью: интерфейс обещает ключ suggestions всегда.
    r = categorize(PRIV_H, "ВкусВилл")
    assert r["via"] == "alias" and r["suggestions"] is None

    class R:
        category = "Продукты"
        confident = True
    monkeypatch.setattr("budget_agent.structured_call", lambda *a, **k: R())
    r = categorize(PRIV_H, "Ашан")
    assert r["via"] == "llm" and r["suggestions"] is None


# ===== Задача 8: MCP-клиент Банка России =====

from budget_agent import MCPClient, cbr_get_rate

def test_mcp_lists_five_tools():
    m = MCPClient()
    try:
        names = {t["name"] for t in m.list_tools()}
    finally:
        m.close()
    assert {"get_rate", "history_rates", "key_rate", "inflation", "statistics"} <= names

def test_cbr_rate_returns_value_and_date():
    r = cbr_get_rate(PRIV_H, "USD")
    assert float(r["data"]["value"]) > 0
    assert r["data"]["date"]
    assert r["source_keys"][0].startswith("cbr:USD@")

def test_mcp_failure_degrades_not_crashes(monkeypatch):
    monkeypatch.setattr("budget_agent.MCPClient.call_tool",
                        lambda self, n, a: (_ for _ in ()).throw(RuntimeError("сеть недоступна")))
    r = cbr_get_rate(PRIV_H, "USD")
    assert r["data"] is None and "error" in r


# ===== Задача 8, раунд ревью 1: четвёртый канал отказа и зависание =====
#
# Находка 1: успешный call_tool с ответом неожиданной формы (сервер сменил
# схему) бросал KeyError мимо try/except, потому что сборка source_keys
# стояла вне защищённого блока. Три теста ниже подменяют call_tool так,
# чтобы он вернул валидный, но неполный словарь — как если бы API сервера
# изменилось, а версия в SETTINGS.mcp_cbr_cmd не совпала с ожиданиями кода.

from budget_agent import cbr_key_rate, cbr_inflation

def test_cbr_get_rate_survives_response_without_expected_field(monkeypatch):
    monkeypatch.setattr("budget_agent.MCPClient.call_tool",
                        lambda self, n, a: {"value": "82.9"})   # нет 'date'
    r = cbr_get_rate(PRIV_H, "USD")
    assert r["data"] is None and "error" in r

def test_cbr_key_rate_survives_response_without_expected_field(monkeypatch):
    monkeypatch.setattr("budget_agent.MCPClient.call_tool",
                        lambda self, n, a: {"points": []})      # нет 'date_to'
    r = cbr_key_rate(PRIV_H)
    assert r["data"] is None and "error" in r

def test_cbr_inflation_survives_response_without_expected_field(monkeypatch):
    monkeypatch.setattr("budget_agent.MCPClient.call_tool",
                        lambda self, n, a: {"points": []})      # нет year_from/year_to
    r = cbr_inflation(PRIV_H, None, None)
    assert r["data"] is None and "error" in r


# Находка 2: без таймаута на чтение ответа зависший сервер блокирует
# однопоточного бота навсегда. "sleep 60" как MCP-сервер не отвечает вообще —
# initialize должен отказать по init_timeout, а не повиснуть на readline().

import sys, time

def test_mcp_init_timeout_does_not_hang():
    start = time.monotonic()
    with pytest.raises(Exception):
        MCPClient(command="sleep 60", init_timeout=0.5, call_timeout=0.5)
    elapsed = time.monotonic() - start
    assert elapsed < 5, "клиент завис на инициализации вместо отказа по таймауту"

# Отдельно проверяем таймаут именно обычного вызова (initialize проходит
# нормально, а зависает уже tools/call) — сервер-заглушка отвечает на
# initialize валидным JSON-RPC и дальше молчит.
def test_mcp_call_timeout_does_not_hang(tmp_path):
    script = tmp_path / "fake_hanging_mcp_server.py"
    script.write_text(
        "import sys, json, time\n"
        "print(json.dumps({'jsonrpc': '2.0', 'id': 1, 'result': {\n"
        "    'protocolVersion': '2024-11-05', 'capabilities': {},\n"
        "    'serverInfo': {'name': 'fake', 'version': '0'}}}), flush=True)\n"
        "sys.stdin.readline()\n"    # notifications/initialized, ответа не требует
        "time.sleep(120)\n"         # дальше сервер не отвечает ни на что
    )
    client = MCPClient(command=f"{sys.executable} {script}", call_timeout=0.5)
    try:
        start = time.monotonic()
        with pytest.raises(Exception):
            client.call_tool("get_rate", {"char_code": "USD"})
        elapsed = time.monotonic() - start
        assert elapsed < 5, "вызов завис вместо отказа по таймауту"
    finally:
        client.close()

def test_mcp_rejects_response_with_wrong_jsonrpc_id(tmp_path):
    script = tmp_path / "fake_wrong_id_mcp_server.py"
    script.write_text(
        "import sys, json\n"
        "for line in sys.stdin:\n"
        "    req = json.loads(line)\n"
        "    if 'id' in req:\n"
        "        print(json.dumps({'jsonrpc': '2.0', 'id': req['id'] + 100, "
        "'result': {'protocolVersion': '2024-11-05', 'capabilities': {}, "
        "'serverInfo': {'name': 'fake', 'version': '0'}}}), flush=True)\n"
    )
    with pytest.raises(RuntimeError, match="JSON-RPC"):
        MCPClient(command=f"{sys.executable} {script}", init_timeout=1)

# ===== Задача 9: SGR-цикл =====

from budget_agent import run_agent, FinalAnswer, NextStep, render_answer

def _script(steps):
    """Подменяет structured_call заранее заданной последовательностью ответов."""
    it = iter(steps)
    return lambda schema, messages, **kw: next(it)

def test_agent_calls_tool_then_finalizes(monkeypatch):
    step = NextStep(goal_progress="начало", plan_remaining_steps=["узнать баланс"],
                    decision_summary="смотрю остатки",
                    call={"tool": "get_balance"}, task_completed=False)
    done = NextStep(goal_progress="готово", plan_remaining_steps=["ответить"],
                    decision_summary="данные есть", call={"tool": "none"}, task_completed=True)
    final = FinalAnswer(summary="На счетах 768 000 рублей.", details=["RUB: 768000"],
                        scenarios=[], source_keys=["family_data"])
    monkeypatch.setattr("budget_agent.structured_call", _script([step, done, final]))
    ans, registry, any_empty_result = run_agent("сколько у нас денег", PRIV_H)
    assert "768" in ans.summary
    assert ans.source_keys == ["family_data"]

def test_agent_drops_invented_sources(monkeypatch):
    step = NextStep(goal_progress="готово", plan_remaining_steps=["ответить"],
                    decision_summary="", call={"tool": "none"}, task_completed=True)
    final = FinalAnswer(summary="ответ", details=[], scenarios=[],
                        source_keys=["PF-999", "family_data"])
    monkeypatch.setattr("budget_agent.structured_call", _script([step, final]))
    ans, registry, any_empty_result = run_agent("вопрос", PRIV_H)
    assert "PF-999" not in ans.source_keys, "источник не из реестра шага должен отброситься"

def test_agent_stops_at_max_steps(monkeypatch):
    loop = NextStep(goal_progress="кручусь", plan_remaining_steps=["ещё"],
                    decision_summary="", call={"tool": "get_balance"}, task_completed=False)
    final = FinalAnswer(summary="не уложился", details=[], scenarios=[], source_keys=[])
    monkeypatch.setattr("budget_agent.structured_call",
                        lambda schema, messages, **kw: final if schema is FinalAnswer else loop)
    ans, registry, any_empty_result = run_agent("вопрос", PRIV_H, max_steps=3)
    assert isinstance(ans, FinalAnswer)

def test_agent_survives_structured_failure(monkeypatch):
    from budget_agent import StructuredFailure
    calls = {"n": 0}
    final = FinalAnswer(summary="частичный ответ", details=[], scenarios=[], source_keys=[])
    def flaky(schema, messages, **kw):
        if schema is FinalAnswer:
            return final
        calls["n"] += 1
        raise StructuredFailure("не разобрал")
    monkeypatch.setattr("budget_agent.structured_call", flaky)
    ans, registry, any_empty_result = run_agent("вопрос", PRIV_H, max_steps=2)
    assert isinstance(ans, FinalAnswer), "контролируемый отказ шага не должен ронять агента"

# ===== Задача 9, раунд ревью 1 =====

def test_cbr_disclaimer_survives_empty_model_sources(monkeypatch):
    """Реальный вызов курса ЦБ; финализация с пустым списком источников —
    оговорка про курс всё равно должна появиться: решается по факту вызова
    инструмента (реестру), а не по тому, что вернула модель."""
    step = NextStep(goal_progress="начало", plan_remaining_steps=["узнать курс"],
                    decision_summary="смотрю курс доллара",
                    call={"tool": "cbr_get_rate", "char_code": "USD"}, task_completed=False)
    done = NextStep(goal_progress="готово", plan_remaining_steps=["ответить"],
                    decision_summary="курс есть", call={"tool": "none"}, task_completed=True)
    final = FinalAnswer(summary="Курс доллара сегодня около 90 рублей.", details=[],
                        scenarios=[], source_keys=[])   # источник не назван моделью
    monkeypatch.setattr("budget_agent.structured_call", _script([step, done, final]))
    ans, registry, any_empty_result = run_agent("какой сегодня курс доллара", PRIV_H)
    text = render_answer(ans, PRIV_H, "какой сегодня курс доллара", registry, any_empty_result)
    assert "Курс приведён по данным ЦБ" in text


@pytest.mark.parametrize("source_key", [
    "cbr:key_rate@2026-08-26",
    "cbr:inflation@2025-2026",
])
def test_cbr_non_currency_sources_do_not_get_exchange_rate_disclaimer(source_key):
    """Ключевая ставка и инфляция — данные ЦБ, но не банковский курс обмена."""
    from budget_agent import needs_disclaimer

    assert needs_disclaimer("Какие сейчас данные ЦБ?", [source_key]) is None

def test_cbr_disclaimer_survives_omitted_source(monkeypatch):
    """Модель называет один источник (family_data), но забывает cbr — список
    непустой, поэтому пункт 4 (подстановка реестра при пустом списке) тут не
    сработает. Оговорка обязана появиться всё равно: она не читает
    source_keys модели, а смотрит на реестр вызванных инструментов."""
    cbr_step = NextStep(goal_progress="начало", plan_remaining_steps=["курс", "баланс"],
                        decision_summary="смотрю курс доллара",
                        call={"tool": "cbr_get_rate", "char_code": "USD"}, task_completed=False)
    balance_step = NextStep(goal_progress="курс есть", plan_remaining_steps=["баланс"],
                            decision_summary="смотрю баланс",
                            call={"tool": "get_balance"}, task_completed=False)
    done = NextStep(goal_progress="готово", plan_remaining_steps=["ответить"],
                    decision_summary="данные есть", call={"tool": "none"}, task_completed=True)
    final = FinalAnswer(summary="Курс доллара сегодня около 90 рублей, на счетах есть деньги.",
                        details=[], scenarios=[], source_keys=["family_data"])   # cbr забыт
    monkeypatch.setattr("budget_agent.structured_call",
                        _script([cbr_step, balance_step, done, final]))
    ans, registry, any_empty_result = run_agent("какой сегодня курс доллара и сколько у нас денег", PRIV_H)
    assert ans.source_keys == ["family_data"], "неполное цитирование модели не форсируем"
    text = render_answer(ans, PRIV_H, "какой сегодня курс доллара и сколько у нас денег", registry, any_empty_result)
    assert "Курс приведён по данным ЦБ" in text, \
        "оговорка не должна зависеть от того, что модель включила в source_keys"

def test_agent_survives_final_structured_failure(monkeypatch):
    from budget_agent import StructuredFailure
    step = NextStep(goal_progress="начало", plan_remaining_steps=["баланс"],
                    decision_summary="смотрю баланс",
                    call={"tool": "get_balance"}, task_completed=False)
    done = NextStep(goal_progress="готово", plan_remaining_steps=["ответить"],
                    decision_summary="данные есть", call={"tool": "none"}, task_completed=True)
    it = iter([step, done])
    def flaky(schema, messages, **kw):
        if schema is FinalAnswer:
            raise StructuredFailure("не разобрал финализацию")
        return next(it)
    monkeypatch.setattr("budget_agent.structured_call", flaky)
    ans, registry, any_empty_result = run_agent("сколько у нас денег", PRIV_H)
    assert isinstance(ans, FinalAnswer), "провал финализации не должен ронять агента"
    assert "family_data" in ans.source_keys, "реестр подставляется, раз модель ничего не сказала"

def test_agent_survives_bad_expense_period(monkeypatch):
    """Неверный формат периода — ValueError внутри get_expenses/_period_range."""
    step = NextStep(goal_progress="начало", plan_remaining_steps=["траты"],
                    decision_summary="смотрю траты",
                    call={"tool": "get_expenses", "period": "весна"}, task_completed=False)
    done = NextStep(goal_progress="готово", plan_remaining_steps=["ответить"],
                    decision_summary="не вышло", call={"tool": "none"}, task_completed=True)
    final = FinalAnswer(summary="не удалось получить траты", details=[], scenarios=[],
                        source_keys=[])
    monkeypatch.setattr("budget_agent.structured_call", _script([step, done, final]))
    ans, registry, any_empty_result = run_agent("сколько мы потратили", PRIV_H)
    assert isinstance(ans, FinalAnswer), "ValueError из инструмента не должен ронять агента"

def test_agent_survives_embed_failure(monkeypatch):
    """Сервис эмбеддингов недоступен — search_knowledge дёргает embed() по
    сети без собственной защиты (в отличие от cbr_*, которые уже отказоустойчивы
    сами по себе); обычная сетевая недоступность не должна ронять агента."""
    step = NextStep(goal_progress="начало", plan_remaining_steps=["поиск"],
                    decision_summary="ищу принцип",
                    call={"tool": "search_knowledge", "query": "бюджет"}, task_completed=False)
    done = NextStep(goal_progress="готово", plan_remaining_steps=["ответить"],
                    decision_summary="не вышло", call={"tool": "none"}, task_completed=True)
    final = FinalAnswer(summary="сервис поиска недоступен", details=[], scenarios=[],
                        source_keys=[])
    monkeypatch.setattr("budget_agent.structured_call", _script([step, done, final]))
    def fake_post(*args, **kwargs):
        raise ConnectionError("сервис эмбеддингов недоступен")
    monkeypatch.setattr("budget_agent.httpx.post", fake_post)
    ans, registry, any_empty_result = run_agent("расскажи про бюджет", PRIV_H)
    assert isinstance(ans, FinalAnswer), "сбой сети у embed() не должен ронять агента"

def test_agent_survives_arbitrary_tool_exception(monkeypatch):
    """Произвольное исключение из инструмента (третий путь, не только у
    периода и эмбеддингов) — тоже не должно ронять цикл."""
    step = NextStep(goal_progress="начало", plan_remaining_steps=["баланс"],
                    decision_summary="смотрю баланс",
                    call={"tool": "get_balance"}, task_completed=False)
    done = NextStep(goal_progress="готово", plan_remaining_steps=["ответить"],
                    decision_summary="не вышло", call={"tool": "none"}, task_completed=True)
    final = FinalAnswer(summary="не удалось получить баланс", details=[], scenarios=[],
                        source_keys=[])
    monkeypatch.setattr("budget_agent.structured_call", _script([step, done, final]))
    monkeypatch.setattr("budget_agent.get_balance",
                        lambda ctx: (_ for _ in ()).throw(RuntimeError("БД недоступна")))
    ans, registry, any_empty_result = run_agent("сколько денег", PRIV_H)
    assert isinstance(ans, FinalAnswer), "произвольное исключение инструмента не должно ронять агента"

# ===== Задача 9, раунд ревью 2 =====

def test_cbr_disclaimer_survives_serialization_roundtrip(monkeypatch):
    """Раунд ревью 2: реестр — теперь явное второе значение возврата
    run_agent, а не приватный атрибут FinalAnswer._used_registry. Атрибут
    тихо терялся при восстановлении FinalAnswer из сериализованного вида
    (model_validate_json после JSON — ровно то, что подключат очередь
    повторов/кэш/восстановление после падения процесса в задачах 10-11),
    откатывая needs_disclaimer к source_keys модели — то есть ровно к
    дефекту, который раунд 1 закрывал. Модель называет только family_data,
    забывая про cbr (список непустой — пункт 4 про подстановку реестра при
    пустом списке тут не сработает), поэтому это тот самый случай, где
    старая реализация тихо теряла оговорку."""
    cbr_step = NextStep(goal_progress="начало", plan_remaining_steps=["курс", "баланс"],
                        decision_summary="смотрю курс доллара",
                        call={"tool": "cbr_get_rate", "char_code": "USD"}, task_completed=False)
    balance_step = NextStep(goal_progress="курс есть", plan_remaining_steps=["баланс"],
                            decision_summary="смотрю баланс",
                            call={"tool": "get_balance"}, task_completed=False)
    done = NextStep(goal_progress="готово", plan_remaining_steps=["ответить"],
                    decision_summary="данные есть", call={"tool": "none"}, task_completed=True)
    final = FinalAnswer(summary="Курс доллара сегодня около 90 рублей, на счетах есть деньги.",
                        details=[], scenarios=[], source_keys=["family_data"])   # cbr забыт
    monkeypatch.setattr("budget_agent.structured_call",
                        _script([cbr_step, balance_step, done, final]))
    ans, registry, any_empty_result = run_agent("какой сегодня курс доллара и сколько у нас денег", PRIV_H)

    # Граница сериализации — ровно то, что подключат задачи 10-11: ans
    # пересекает её как JSON, registry несётся рядом явным аргументом.
    restored = FinalAnswer.model_validate_json(ans.model_dump_json())
    assert restored.source_keys == ["family_data"], "cbr в цитировании по-прежнему нет — ожидаемо"

    text = render_answer(restored, PRIV_H, "какой сегодня курс доллара и сколько у нас денег",
                         registry, any_empty_result)
    assert "Курс приведён по данным ЦБ" in text, \
        "оговорка обязана остаться на месте после сериализации FinalAnswer, раз реестр передан явно"

# ===== Задача 10: быстрый путь — команды и отчёты =====
#
# Ctx.person_id был осознанно убран в задаче 6 (мёртвое, ненадёжное поле —
# раунд ревью 1, коммит 55ebd83): у Ctx три поля (person, chat_type,
# chat_id), не четыре, как в первоначальном примере брифа задачи 10. Тесты
# ниже используют актуальную сигнатуру.
#
# Финальная волна, пункт 3: сид-данные (make_seed.py) теперь покрывают и
# неполный текущий календарный месяц (ANCHOR_DATE = date.today(), см.
# докстринг make_seed.py) — period="this_month" в живой базе больше не
# пуст. Но какие именно категории и суммы туда попадут — рандомизировано и
# зависит от дня месяца на момент генерации, поэтому тесты, которым нужна
# ПРЕДСКАЗУЕМАЯ трата текущего месяца (конкретная категория/сумма), всё
# равно заводят её сами через add_expense (без обращения к модели, тот же
# приём, что и в тестах задачи 7) и убирают за собой в finally/фикстуре —
# так тест не зависит от случайного содержимого сгенерированного месяца.

from budget_agent import (handle_command, render_status, render_report,
                          HELP_TEXT, daily_report, send_scheduled_report, SETTINGS)

def test_status_shows_spent_limit_percent():
    txt = render_status(PRIV_H)
    assert "из" in txt and "%" in txt

def test_start_sets_awaiting_limit():
    handle_command("/start", Ctx("husband", "private", 77), None)
    try:
        assert get_state(77)["state"] == "awaiting_limit"
    finally:
        with db() as conn:
            conn.execute("DELETE FROM dialog_state WHERE chat_id=%s", (77,))

@pytest.fixture
def this_month_products_expense():
    r = add_expense(PRIV_H, 500, "Тестовый Продуктовый", category="Продукты", scope="husband")
    yield
    with db() as conn:
        conn.execute("DELETE FROM transactions WHERE id=%s", (r["data"]["id"],))

def test_report_returns_keyboard_with_advice_button(this_month_products_expense):
    txt, markup = render_report(PRIV_H)
    assert "Продукты" in txt
    buttons = markup["inline_keyboard"][0]
    assert buttons[0]["callback_data"] == "ai_advice"

def test_report_empty_month_says_no_expenses_yet(monkeypatch):
    # Финальная волна, пункт 7: пустой this_month раньше давал заголовок
    # "Расходы за месяц:" без единой строки под ним — читалось как
    # оборванный вывод. Сид теперь почти всегда покрывает текущий месяц
    # (пункт 3), поэтому пустоту здесь имитируем явно через get_expenses,
    # а не полагаемся на то, что в базе пусто от природы.
    # Заглушка повторяет реальный контракт целиком: с появлением
    # other_currencies (ревью PR #20) неполный словарь означал бы, что тест
    # зеленеет на форме, которой инструмент уже не возвращает.
    monkeypatch.setattr("budget_agent.get_expenses",
                        lambda ctx, period, category: {
                            "data": {"by_category": [], "other_currencies": []}})
    txt, markup = render_report(PRIV_H)
    assert txt == "В этом месяце трат ещё не было."
    assert markup["inline_keyboard"][0][0]["callback_data"] == "ai_advice"

def test_commands_do_not_call_llm(monkeypatch):
    monkeypatch.setattr("budget_agent.structured_call",
                        lambda *a, **k: pytest.fail("команды идут без модели"))
    render_status(PRIV_H)
    render_report(PRIV_H)

@pytest.fixture
def personal_category_this_month_expense():
    r = add_expense(PRIV_H, 3000, "Тестовый Бутик", category="Одежда", scope="husband")
    yield
    with db() as conn:
        conn.execute("DELETE FROM transactions WHERE id=%s", (r["data"]["id"],))

def test_report_in_group_hides_personal_categories(personal_category_this_month_expense):
    # Заменено по указанию ревьюера: сравнение длин строк проходит
    # тривиально и ничего не гарантирует (пустой отчёт короче непустого по
    # множеству причин, не только из-за приватности). Проверяем содержательно:
    # личная категория ("Одежда", заведена только с scope='husband') присутствует
    # в личном отчёте мужа и отсутствует в отчёте для общего чата.
    group_txt, _ = render_report(GROUP)
    priv_txt, _ = render_report(PRIV_H)
    assert "Одежда" in priv_txt, "личная категория должна быть видна в личном отчёте"
    assert "Одежда" not in group_txt, "личная категория не должна попасть в общий отчёт"


def test_daily_report_covers_yesterday_only():
    txt = daily_report(GROUP)
    assert "вчера" in txt.lower() or "за вчера" in txt.lower()

def test_daily_report_uses_common_scope_only(monkeypatch):
    # Рассылка идёт в общий чат, личные траты в неё попасть не должны.
    #
    # Заменено по сравнению с брифом: там подмена была однострочной лямбдой
    # `seen.setdefault("ctx", ctx) or {...}` — dict.setdefault возвращает
    # сохранённое значение (сам объект ctx), а он всегда truthy как обычный
    # объект без __bool__, поэтому `or` возвращал ctx вместо словаря с
    # "data", и daily_report падал с TypeError на `["data"]`. Обычная
    # функция с явным телом делает то же самое без этой ловушки.
    #
    # Раунд ревью 1: вызов с GROUP ничего не проверял — GROUP сам по себе
    # (chat_type="group") уже даёт visible_scopes == ("common",) независимо
    # от того, нормализует daily_report ctx внутри себя или просто
    # форвардит его дальше. Ревьюер откатил нормализацию и прогнал тест с
    # GROUP — тест остался зелёным, то есть не отличал защищённый код от
    # незащищённого. Вызываем с PRIV_H (person="husband", chat_type="private")
    # — это ровно тот контекст, при котором утечка происходит без защиты:
    # без принудительной нормализации внутри daily_report get_expenses
    # получил бы visible_scopes("husband","private") == ("common","husband"),
    # и личные траты мужа ушли бы в общую рассылку.
    seen = {}
    def fake_get_expenses(ctx, period, category):
        seen["ctx"] = ctx
        return {"data": {"period": period, "category": None, "total": 0.0,
                         "by_category": []}, "source_keys": ["family_data"]}
    monkeypatch.setattr("budget_agent.get_expenses", fake_get_expenses)
    daily_report(PRIV_H)
    assert visible_scopes(seen["ctx"].person, seen["ctx"].chat_type) == ("common",)

def test_scheduled_report_without_token_does_not_crash(monkeypatch):
    monkeypatch.setattr("budget_agent.SETTINGS", SETTINGS.__class__(
        **{**SETTINGS.__dict__, "telegram_token": None}))
    send_scheduled_report()   # должен просто напечатать отчёт в stdout

def test_scheduled_report_propagates_telegram_http_error(monkeypatch):
    monkeypatch.setattr("budget_agent.SETTINGS", SETTINGS.__class__(
        **{**SETTINGS.__dict__, "telegram_token": "fake"}))
    monkeypatch.setenv("TELEGRAM_REPORT_CHAT_ID", "123")
    monkeypatch.setattr("budget_agent.daily_report", lambda ctx: "report")
    class FailedResponse:
        def raise_for_status(self):
            raise RuntimeError("telegram unavailable")
    monkeypatch.setattr("budget_agent.httpx.post", lambda *a, **k: FailedResponse())
    with pytest.raises(RuntimeError, match="telegram unavailable"):
        send_scheduled_report()


# ===== Задача 11: Telegram-слой и маршрутизация =====
#
# Брифовые тесты (шаг 1) переписаны под фактические сигнатуры, зафиксированные
# рулингами задачи 9: Ctx трёхпольный (без person_id — брифовый пример
# `Ctx("husband", "private", 88, 1)` устарел), render_answer принимает
# (ans, ctx, question, registry), run_agent возвращает пару (ans, registry).

import asyncio
import logging
from budget_agent import (route_message, resolve_ctx, run_agent, FinalAnswer,
                          _try_complete_limit, _try_complete_category, _handle_spending,
                          _on_text, _on_advice, run_telegram, ParsedTransaction,
                          _ON_TEXT_ERROR_MSG, _ON_ADVICE_ERROR_MSG,
                          _is_authorized_ctx, _UNAUTHORIZED_MSG)

def _settings_with_persons(**overrides):
    base = {"husband_tg_id": 555, "wife_tg_id": 666}
    base.update(overrides)
    return SETTINGS.__class__(**{**SETTINGS.__dict__, **base})


# ---- Шаг 1: порядок разбора сообщения ----

def test_command_goes_to_fast_path(monkeypatch):
    monkeypatch.setattr("budget_agent.run_agent",
                        lambda *a, **k: pytest.fail("команда не идёт в SGR"))
    assert "из" in route_message("/status", PRIV_H)

def test_spending_text_goes_to_add_expense(monkeypatch):
    monkeypatch.setattr("budget_agent.run_agent",
                        lambda *a, **k: pytest.fail("трата не идёт в SGR"))
    # Реальный (не подменённый) вызов categorize/parse_transaction: модель иногда
    # нормализует падеж мерчанта ("в Пятёрочке" -> merchant="Пятёрочка"), поэтому
    # чистим по id новой строки, а не по точному тексту мерчанта — иначе уборка
    # молча промахивается и оставляет хвост в общей базе.
    with db() as conn:
        before_max = conn.execute("SELECT COALESCE(MAX(id), 0) FROM transactions").fetchone()[0]
    out = route_message("потратил 850 в Пятёрочке", PRIV_H)
    try:
        assert "Продукты" in out and "850" in out
    finally:
        with db() as conn:
            conn.execute("DELETE FROM transactions WHERE id > %s", (before_max,))

def test_question_goes_to_agent(monkeypatch):
    # Брифовый вариант `called.setdefault("q", q) or FinalAnswer(...)` содержит
    # ровно ту ловушку, о которой уже предупреждает test_daily_report_covers_...
    # выше по файлу: dict.setdefault возвращает сохранённый q, а он truthy как
    # любая непустая строка, поэтому `or` вернул бы q вместо кортежа. Обычная
    # функция с явным телом её не имеет.
    called = {}
    def fake_run_agent(q, ctx, **k):
        called["q"] = q
        return FinalAnswer(summary="ответ", details=[], scenarios=[], source_keys=[]), set(), False
    monkeypatch.setattr("budget_agent.run_agent", fake_run_agent)
    monkeypatch.setattr("budget_agent.parse_transaction", lambda *a, **k: None)
    out = route_message("сначала гасить кредит или копить подушку?", PRIV_H)
    assert "кредит" in called["q"]
    assert "ответ" in out

def test_question_with_amount_does_not_go_to_add_expense(monkeypatch):
    # Раунд ревью 2 задачи 12: до правки классификатор (_SpendingCheck)
    # судил о трате по наличию суммы в тексте — вопрос «Отпуск обойдётся в
    # 280 тысяч — тянем?» в 4 из 6 живых прогонов уходил в add_expense
    # вместо агента (demo_scenarios.md, сценарий 7). Реальный (не
    # подменённый) вызов parse_transaction — тест бессмыслен с замоканным
    # классификатором, он обязан бить по промпту, который чинили, а не по
    # обвязке вокруг него. run_agent подменён только затем, чтобы не ждать
    # полный SGR-цикл — сама проверка про то, что вопрос вообще дошёл сюда,
    # а не был перехвачен как трата раньше по цепочке route_message.
    monkeypatch.setattr(
        "budget_agent.run_agent",
        lambda q, ctx, **k: (
            FinalAnswer(summary="ответ агента", details=[], scenarios=[], source_keys=[]),
            set(), False))
    with db() as conn:
        before_max = conn.execute("SELECT COALESCE(MAX(id), 0) FROM transactions").fetchone()[0]
    try:
        out = route_message("Отпуск обойдётся в 280 тысяч — тянем?", PRIV_H)
        assert "ответ агента" in out, \
            "вопрос с суммой не должен перехватываться parse_transaction как трата"
        with db() as conn:
            after_max = conn.execute(
                "SELECT COALESCE(MAX(id), 0) FROM transactions").fetchone()[0]
        assert after_max == before_max, "вопрос не должен создавать запись о трате"
    finally:
        with db() as conn:
            conn.execute("DELETE FROM transactions WHERE id > %s", (before_max,))

def test_awaiting_category_answer_completes_pending():
    set_state(88, "awaiting_category", {"amount": 4100, "merchant": "Ашан"})
    ctx = Ctx("husband", "private", 88)
    try:
        out = route_message("Продукты", ctx)
        assert "4100" in out.replace(" ", "") or "4 100" in out
        assert get_state(88)["state"] == "base"
    finally:
        with db() as conn:
            conn.execute("DELETE FROM transactions WHERE merchant='Ашан'")
            conn.execute("DELETE FROM dialog_state WHERE chat_id=88")
            conn.execute("DELETE FROM merchant_aliases WHERE alias='ашан' AND scope='common'")

def test_resolve_ctx_maps_telegram_id_to_person(monkeypatch):
    monkeypatch.setattr("budget_agent.SETTINGS", _settings_with_persons())
    class U:
        class message:
            chat_id = 555
            class chat: type = "private"
            class from_user: id = 555
    assert resolve_ctx(U).person == "husband"


# ---- Порядок обязателен: тесты, доказывающие каждую границу отдельно ----

def test_command_bypasses_dialog_state(monkeypatch):
    # Если бы состояние проверялось раньше команды, "/status" ушёл бы в
    # _complete_limit как текст ответа на лимит — команда обязана
    # перехватываться до всякого обращения к dialog_state.
    monkeypatch.setattr("budget_agent.run_agent",
                        lambda *a, **k: pytest.fail("не должно быть SGR"))
    set_state(90, "awaiting_limit")
    ctx = Ctx("husband", "private", 90)
    try:
        out = route_message("/status", ctx)
        assert "из" in out
        assert get_state(90)["state"] == "awaiting_limit", \
            "команда не должна завершать висящее состояние диалога"
    finally:
        with db() as conn:
            conn.execute("DELETE FROM dialog_state WHERE chat_id=90")

def test_awaiting_state_intercepts_before_spending_parse(monkeypatch):
    # Риск, названный в задаче: если бы трата проверялась раньше состояния,
    # ответ на висящий вопрос мог случайно распарситься как новая трата и
    # уйти мимо ожидаемого пути. parse_transaction здесь роняет тест, если его
    # вообще вызвали — только это и доказывает нужный порядок.
    monkeypatch.setattr("budget_agent.parse_transaction",
                        lambda *a, **k: pytest.fail("трата не должна разбираться, "
                                                     "пока есть висящий вопрос"))
    set_state(89, "awaiting_category", {"amount": 500, "merchant": "ТЕСТ_Такси_89"})
    ctx = Ctx("husband", "private", 89)
    try:
        out = route_message("Транспорт", ctx)
        assert "Транспорт" in out
        assert get_state(89)["state"] == "base"
    finally:
        with db() as conn:
            conn.execute("DELETE FROM transactions WHERE merchant='ТЕСТ_Такси_89'")
            conn.execute("DELETE FROM dialog_state WHERE chat_id=89")
            conn.execute("DELETE FROM merchant_aliases WHERE alias='тест_такси_89' AND scope='common'")

def test_awaiting_limit_intercepts_before_spending_parse(monkeypatch):
    monkeypatch.setattr("budget_agent.parse_transaction",
                        lambda *a, **k: pytest.fail("трата не должна разбираться, "
                                                     "пока ждём ответ про лимит"))
    set_state(91, "awaiting_limit")
    ctx = Ctx("husband", "private", 91)
    try:
        out = route_message("50000", ctx)
        assert "50 000" in out or "50000" in out
        assert get_state(91)["state"] == "base"
    finally:
        with db() as conn:
            conn.execute("DELETE FROM dialog_state WHERE chat_id=91")
            conn.execute("DELETE FROM budget_limits WHERE period='monthly' AND scope='common'")
        # восстанавливаем канонический сид-лимит, который затёрли выше
        with db() as conn:
            conn.execute(
                "INSERT INTO budget_limits (period, amount, currency, scope) "
                "VALUES ('monthly', 140000, 'RUB', 'common') "
                "ON CONFLICT (period, scope) DO UPDATE SET amount = EXCLUDED.amount")


# ---- Незавершённый ввод: не роняет бота ----
#
# _try_complete_limit/_try_complete_category возвращают None вместо "reask",
# когда текст не похож на ответ — низкоуровневая проверка самих функций.

def test_try_complete_limit_returns_none_without_digits():
    out = _try_complete_limit("не знаю", Ctx("husband", "private", 92))
    assert out is None

def test_try_complete_category_returns_none_on_unknown_name():
    out = _try_complete_category("Криптовалюта", Ctx("husband", "private", 93),
                                 {"amount": 100, "merchant": "ТЕСТ_Мерчант_93"})
    assert out is None
    with db() as conn:
        row = conn.execute("SELECT 1 FROM transactions WHERE merchant='ТЕСТ_Мерчант_93'").fetchone()
    assert row is None, "неизвестная категория не должна создавать трату"

def test_try_complete_category_case_insensitive_match():
    ctx = Ctx("husband", "private", 94)
    try:
        out = _try_complete_category("продукты", ctx, {"amount": 200, "merchant": "ТЕСТ_Мерчант_94"})
        assert out is not None and "Продукты" in out
    finally:
        with db() as conn:
            conn.execute("DELETE FROM transactions WHERE merchant='ТЕСТ_Мерчант_94'")
            conn.execute("DELETE FROM merchant_aliases WHERE alias='тест_мерчант_94' AND scope='common'")
            conn.execute("DELETE FROM dialog_state WHERE chat_id=94")


# ---- Раунд ревью 1: свободный вопрос в состоянии ожидания сбрасывает pending ----
#
# Ревьюер воспроизвёл живьём: бот спрашивает категорию, пользователь задаёт
# другой вопрос ("сколько мы потратили на еду?") — и вместо ответа получает
# переспрос категории, свой вопрос вообще не долетает до цикла. Требование
# design-doc (раздел «Состояние диалога»): свободный вопрос сбрасывает
# pending, а не удерживается состоянием бесконечно (до истечения
# STATE_TTL_SECONDS). Порядок ветвей route_message при этом не меняется —
# состояние по-прежнему проверяется раньше траты и раньше SGR, дело в том,
# что происходит ВНУТРИ шага состояния при несовпадении.

def test_awaiting_category_free_question_resets_and_answers(monkeypatch):
    set_state(96, "awaiting_category", {"amount": 500, "merchant": "ТЕСТ_Мерчант_96"})
    ctx = Ctx("husband", "private", 96)
    called = {}
    def fake_run_agent(q, ctx, on_step=None, max_steps=8):
        called["q"] = q
        return FinalAnswer(summary="на еду в этом месяце потрачено 12000 ₽",
                           details=[], scenarios=[], source_keys=[]), set(), False
    monkeypatch.setattr("budget_agent.run_agent", fake_run_agent)
    monkeypatch.setattr("budget_agent.parse_transaction", lambda *a, **k: None)
    try:
        out = route_message("сколько мы потратили на еду?", ctx)
        assert "еду" in called["q"], "свой вопрос пользователя должен дойти до SGR-цикла"
        assert "на еду в этом месяце потрачено 12000" in out, "ответ на свой вопрос должен прийти"
        assert "не сохранена" in out, "пользователь должен узнать, что трата потеряна"
        assert get_state(96)["state"] == "base"
        with db() as conn:
            row = conn.execute("SELECT 1 FROM transactions WHERE merchant='ТЕСТ_Мерчант_96'").fetchone()
        assert row is None, "незавершённая трата не должна была создаться"
    finally:
        with db() as conn:
            conn.execute("DELETE FROM dialog_state WHERE chat_id=96")

def test_awaiting_limit_free_question_resets_and_answers(monkeypatch):
    set_state(97, "awaiting_limit")
    ctx = Ctx("husband", "private", 97)
    def fake_run_agent(q, ctx, on_step=None, max_steps=8):
        return FinalAnswer(summary="подушки хватит на два месяца", details=[],
                           scenarios=[], source_keys=[]), set(), False
    monkeypatch.setattr("budget_agent.run_agent", fake_run_agent)
    monkeypatch.setattr("budget_agent.parse_transaction", lambda *a, **k: None)
    try:
        out = route_message("хватит ли нам денег до зарплаты?", ctx)
        assert "подушки хватит на два месяца" in out
        assert get_state(97)["state"] == "base"
    finally:
        with db() as conn:
            conn.execute("DELETE FROM dialog_state WHERE chat_id=97")

def test_awaiting_category_real_answer_still_completes(monkeypatch):
    # Регрессия: настоящий ответ категорией по-прежнему создаёт трату и не
    # уходит в SGR — сброс касается только нераспознанных ответов.
    set_state(98, "awaiting_category", {"amount": 640, "merchant": "ТЕСТ_Мерчант_98"})
    ctx = Ctx("husband", "private", 98)
    monkeypatch.setattr("budget_agent.run_agent",
                        lambda *a, **k: pytest.fail("настоящий ответ категорией не должен идти в SGR"))
    try:
        out = route_message("Продукты", ctx)
        assert "Продукты" in out and "640" in out
        assert "не сохранена" not in out
        assert get_state(98)["state"] == "base"
        with db() as conn:
            row = conn.execute("SELECT 1 FROM transactions WHERE merchant='ТЕСТ_Мерчант_98'").fetchone()
        assert row is not None
    finally:
        with db() as conn:
            conn.execute("DELETE FROM transactions WHERE merchant='ТЕСТ_Мерчант_98'")
            conn.execute("DELETE FROM dialog_state WHERE chat_id=98")
            conn.execute("DELETE FROM merchant_aliases WHERE alias='тест_мерчант_98' AND scope='common'")

def test_awaiting_limit_real_answer_still_completes():
    set_state(99, "awaiting_limit")
    ctx = Ctx("husband", "private", 99)
    try:
        out = route_message("60000", ctx)
        assert "60 000" in out or "60000" in out
        assert "вернёмся позже" not in out
        assert get_state(99)["state"] == "base"
    finally:
        with db() as conn:
            conn.execute("DELETE FROM dialog_state WHERE chat_id=99")
            conn.execute(
                "INSERT INTO budget_limits (period, amount, currency, scope) "
                "VALUES ('monthly', 140000, 'RUB', 'common') "
                "ON CONFLICT (period, scope) DO UPDATE SET amount = EXCLUDED.amount")

def test_awaiting_category_new_spending_resets_and_records_new(monkeypatch):
    # Пограничный случай: вместо ответа на категорию пользователь присылает
    # НОВУЮ трату — тоже "передумал", но не вопрос, а другое действие.
    # Старая трата теряется (с уведомлением), новая записывается.
    set_state(101, "awaiting_category", {"amount": 300, "merchant": "ТЕСТ_Мерчант_101_old"})
    ctx = Ctx("husband", "private", 101)
    monkeypatch.setattr("budget_agent.parse_transaction",
                        lambda *a, **k: ParsedTransaction(kind="expense", amount=500, counterparty="ТЕСТ_Мерчант_101_new",
                                                       currency="RUB"))
    monkeypatch.setattr("budget_agent.categorize",
                        lambda ctx, merchant: {"category": "Транспорт", "via": "alias", "suggestions": None})
    try:
        out = route_message("потратил 500 в новом месте", ctx)
        assert "не сохранена" in out
        assert "Транспорт" in out and "500" in out
        assert get_state(101)["state"] == "base"
        with db() as conn:
            old = conn.execute("SELECT 1 FROM transactions WHERE merchant='ТЕСТ_Мерчант_101_old'").fetchone()
            new = conn.execute("SELECT 1 FROM transactions WHERE merchant='ТЕСТ_Мерчант_101_new'").fetchone()
        assert old is None, "старая (нераспознанная) трата не должна была создаться"
        assert new is not None, "новая трата должна быть записана"
    finally:
        with db() as conn:
            conn.execute("DELETE FROM transactions WHERE merchant='ТЕСТ_Мерчант_101_new'")
            conn.execute("DELETE FROM dialog_state WHERE chat_id=101")


# ---- resolve_ctx: и message, и callback_query, chat_type без нормализации ----

class _FakeUser:
    def __init__(self, id): self.id = id

class _FakeChat:
    def __init__(self, type): self.type = type

class _FakeMessage:
    def __init__(self, text, chat_id=1, user_id=555, chat_type="private"):
        self.text = text
        self.chat_id = chat_id
        self.chat = _FakeChat(chat_type)
        self.from_user = _FakeUser(user_id)

class _FakeUpdate:
    def __init__(self, text, **kw):
        self.message = _FakeMessage(text, **kw)
        self.callback_query = None

class _FakeCallbackQuery:
    def __init__(self, message, user):
        self.message = message
        self.from_user = user
        self.answered = False
    async def answer(self):
        self.answered = True

class _FakeCQUpdate:
    def __init__(self, user_id=555, chat_id=1, chat_type="private"):
        self.message = None
        self.callback_query = _FakeCallbackQuery(
            _FakeMessage("", chat_id=chat_id, user_id=user_id, chat_type=chat_type),
            _FakeUser(user_id))

def test_resolve_ctx_unknown_user_has_no_person(monkeypatch):
    monkeypatch.setattr("budget_agent.SETTINGS", _settings_with_persons())
    ctx = resolve_ctx(_FakeUpdate("привет", user_id=999))
    assert ctx.person is None

def test_authorization_rejects_unknown_user_and_unlisted_group(monkeypatch):
    monkeypatch.setattr("budget_agent.SETTINGS", _settings_with_persons(
        allowed_group_chat_ids=(-1001,)))
    assert not _is_authorized_ctx(Ctx(None, "private", 1, user_id=999))
    assert not _is_authorized_ctx(Ctx("husband", "group", -2002, user_id=555))
    assert _is_authorized_ctx(Ctx("husband", "group", -1001, user_id=555))

def test_resolve_ctx_does_not_normalize_chat_type(monkeypatch):
    # Тип чата передаётся как есть — защита от приведения списком значений
    # лежит в visible_scopes(), а не здесь (см. брифовое требование раздела).
    monkeypatch.setattr("budget_agent.SETTINGS", _settings_with_persons())
    ctx = resolve_ctx(_FakeUpdate("привет", chat_type="supergroup"))
    assert ctx.chat_type == "supergroup"

def test_resolve_ctx_handles_callback_query(monkeypatch):
    monkeypatch.setattr("budget_agent.SETTINGS", _settings_with_persons())
    ctx = resolve_ctx(_FakeCQUpdate(user_id=666, chat_id=9))
    assert ctx.person == "wife" and ctx.chat_id == 9


# ---- Telegram-обработчики на подменённых обновлениях (без токена и сети) ----

class _FakeBot:
    def __init__(self):
        self.sent = []
    async def send_message(self, chat_id, text, reply_markup=None, parse_mode=None):
        self.sent.append({"chat_id": chat_id, "text": text,
                          "reply_markup": reply_markup, "parse_mode": parse_mode})

class _FakeContext:
    def __init__(self):
        self.bot = _FakeBot()

def test_on_text_sends_command_result(monkeypatch):
    monkeypatch.setattr("budget_agent.SETTINGS", _settings_with_persons(broadcast_steps=False))
    update, context = _FakeUpdate("/status"), _FakeContext()
    asyncio.run(_on_text(update, context))
    assert len(context.bot.sent) == 1
    assert "из" in context.bot.sent[0]["text"]

def test_on_text_rejects_unknown_user_before_routing(monkeypatch):
    monkeypatch.setattr("budget_agent.SETTINGS", _settings_with_persons(broadcast_steps=False))
    monkeypatch.setattr("budget_agent.route_message",
                        lambda *a, **k: pytest.fail("unauthorized request reached routing"))
    update, context = _FakeUpdate("сколько денег", user_id=999), _FakeContext()
    asyncio.run(_on_text(update, context))
    assert context.bot.sent[0]["text"] == _UNAUTHORIZED_MSG

@pytest.mark.parametrize("amount", [0, -1, float("nan"), float("inf")])
def test_parsed_spending_rejects_non_positive_or_non_finite_amount(amount):
    with pytest.raises(Exception):
        ParsedTransaction(kind="expense", amount=amount, counterparty="x", currency="RUB")

def test_on_text_report_attaches_keyboard(monkeypatch):
    monkeypatch.setattr("budget_agent.SETTINGS", _settings_with_persons(broadcast_steps=False))
    update, context = _FakeUpdate("/report"), _FakeContext()
    asyncio.run(_on_text(update, context))
    assert context.bot.sent[-1]["reply_markup"] is not None

def test_on_text_broadcasts_steps_when_enabled(monkeypatch):
    monkeypatch.setattr("budget_agent.SETTINGS", _settings_with_persons(broadcast_steps=True))
    class FakeStep:
        decision_summary = "проверяю баланс"
    def fake_run_agent(question, ctx, on_step=None, max_steps=8):
        if on_step:
            on_step(FakeStep(), 1)
        return (FinalAnswer(summary="итоговый ответ", details=[], scenarios=[], source_keys=[]),
                set(), False)
    monkeypatch.setattr("budget_agent.run_agent", fake_run_agent)
    update, context = _FakeUpdate("сколько у нас денег"), _FakeContext()
    asyncio.run(_on_text(update, context))
    texts = [m["text"] for m in context.bot.sent]
    assert any("проверяю баланс" in t for t in texts), "шаг агента должен уйти отдельным сообщением"
    assert any("итоговый ответ" in t for t in texts), "финальный ответ должен прийти отдельно"
    assert len(texts) >= 2

def test_on_text_no_broadcast_when_disabled(monkeypatch):
    monkeypatch.setattr("budget_agent.SETTINGS", _settings_with_persons(broadcast_steps=False))
    class FakeStep:
        decision_summary = "проверяю баланс"
    def fake_run_agent(question, ctx, on_step=None, max_steps=8):
        if on_step:
            pytest.fail("on_step не должен вызываться, когда трансляция выключена")
        return (FinalAnswer(summary="итоговый ответ", details=[], scenarios=[], source_keys=[]),
                set(), False)
    monkeypatch.setattr("budget_agent.run_agent", fake_run_agent)
    update, context = _FakeUpdate("сколько у нас денег"), _FakeContext()
    asyncio.run(_on_text(update, context))
    texts = [m["text"] for m in context.bot.sent]
    assert texts == ["итоговый ответ"]

# ---- Раунд ревью 1: исключение не должно утекать в чат ----
#
# Ревьюер показал, что сбой подключения к базе кладёт в str(e) хост, порт и
# имя пользователя — ровно те данные, которые прячутся в окружение, а не в
# код. Пользователю — общая формулировка, трассировка — в лог процесса.

_FAKE_DSN_LEAK = "connection failed: postgresql://fakeuser:fakepass000@10.0.0.99:5433/notreal"

def test_on_text_unexpected_exception_replies_without_leaking_internals(monkeypatch, caplog):
    monkeypatch.setattr("budget_agent.SETTINGS", _settings_with_persons(broadcast_steps=False))
    def boom(*a, **k):
        raise RuntimeError(_FAKE_DSN_LEAK)
    monkeypatch.setattr("budget_agent.run_agent", boom)
    monkeypatch.setattr("budget_agent.parse_transaction", lambda *a, **k: None)
    update, context = _FakeUpdate("сколько у нас денег"), _FakeContext()
    with caplog.at_level(logging.ERROR, logger="budget_agent.telegram"):
        asyncio.run(_on_text(update, context))   # не должно выбросить исключение
    sent_texts = [m["text"] for m in context.bot.sent]
    assert sent_texts == [_ON_TEXT_ERROR_MSG]
    for leak in ("10.0.0.99", "5433", "fakeuser", "fakepass000", "notreal"):
        assert leak not in sent_texts[0], f"{leak!r} не должно быть в ответе пользователю"
    # трассировка не потеряна — ушла в лог процесса, а не только в чат
    assert "RuntimeError" in caplog.text
    assert _FAKE_DSN_LEAK in caplog.text

def test_on_advice_unexpected_exception_replies_without_leaking_internals(monkeypatch, caplog):
    monkeypatch.setattr("budget_agent.SETTINGS", _settings_with_persons(broadcast_steps=False))
    def boom(*a, **k):
        raise RuntimeError(_FAKE_DSN_LEAK)
    monkeypatch.setattr("budget_agent.run_agent", boom)
    update, context = _FakeCQUpdate(), _FakeContext()
    with caplog.at_level(logging.ERROR, logger="budget_agent.telegram"):
        asyncio.run(_on_advice(update, context))
    sent_texts = [m["text"] for m in context.bot.sent]
    assert sent_texts == [_ON_ADVICE_ERROR_MSG]
    for leak in ("10.0.0.99", "5433", "fakeuser", "fakepass000"):
        assert leak not in sent_texts[0]
    assert "RuntimeError" in caplog.text

def test_on_advice_answers_callback_and_sends_reply(monkeypatch):
    monkeypatch.setattr("budget_agent.SETTINGS", _settings_with_persons(broadcast_steps=False))
    def fake_run_agent(question, ctx, on_step=None, max_steps=8):
        return (FinalAnswer(summary="совет по бюджету", details=[], scenarios=[], source_keys=[]),
                set(), False)
    monkeypatch.setattr("budget_agent.run_agent", fake_run_agent)
    update, context = _FakeCQUpdate(), _FakeContext()
    asyncio.run(_on_advice(update, context))
    assert update.callback_query.answered
    assert any("совет по бюджету" in m["text"] for m in context.bot.sent)


# ---- run_telegram(): без токена не пытается поллить сеть ----

def test_run_telegram_without_token_returns_cleanly(monkeypatch, capsys):
    monkeypatch.setattr("budget_agent.SETTINGS", SETTINGS.__class__(
        **{**SETTINGS.__dict__, "telegram_token": None}))
    run_telegram()   # не должно кидать исключение и не должно лезть в сеть
    assert "TELEGRAM_BOT_TOKEN" in capsys.readouterr().out


# ===== Правки по живой обратной связи (2026-08-25) =====
#
# 1. Разбор суммы лимита молча искажает значение: re.sub(r"[^\d]", "", text)
#    выбрасывал единицы измерения вместе с пробелами — "250 тыс руб"
#    разбиралось в 250, а не в 250000. _parse_amount ниже — детерминированный
#    разбор с учётом единиц (тыс/к/k → ×1000, млн/м → ×1000000), без
#    обращения к модели, тестируется здесь напрямую как чистая функция.
# 2. Агент не мог менять лимит — только рассуждать. Добавлен инструмент
#    set_budget_limit (по образцу остальных инструментов раздела 5),
#    зарегистрированный в TOOL_REGISTRY/ToolCall и доступный из SGR-цикла.
# 3. Команды бота не регистрировались в меню Telegram — добавлен
#    _post_init (Application.builder().post_init(...)), вызывающий
#    Bot.set_my_commands при старте.

from budget_agent import _parse_amount, set_budget_limit, _post_init, BOT_COMMANDS, NextStep

_CANONICAL_SEED_LIMIT = 140000

def _restore_seed_limit():
    with db() as conn:
        conn.execute(
            "INSERT INTO budget_limits (period, amount, currency, scope) "
            "VALUES ('monthly', %s, 'RUB', 'common') "
            "ON CONFLICT (period, scope) DO UPDATE SET amount = EXCLUDED.amount",
            (_CANONICAL_SEED_LIMIT,))

def _current_limit_row():
    with db() as conn:
        return conn.execute(
            "SELECT amount FROM budget_limits WHERE period='monthly' AND scope='common'"
        ).fetchone()


# ---- 1a. _parse_amount как чистая функция: единицы измерения, разделители ----

@pytest.mark.parametrize("text,expected", [
    ("250 тыс руб", 250000),      # исходный баг из фидбэка
    ("250к", 250000),
    ("250k", 250000),
    ("1,5 млн", 1500000),         # запятая как десятичный разделитель
    ("1.5 млн", 1500000),         # точка как десятичный разделитель
    ("250 000", 250000),          # пробел как разделитель разрядов — уже работало верно
    ("140000 рублей", 140000),    # уже работало верно
    ("300 тысяч", 300000),
    ("2 миллиона", 2000000),
    ("60000", 60000),
])
def test_parse_amount_handles_units_and_separators(text, expected):
    assert _parse_amount(text) == expected

@pytest.mark.parametrize("text", ["примерно столько же", "не знаю", "", "   ", "как обычно"])
def test_parse_amount_returns_none_when_no_number(text):
    assert _parse_amount(text) is None


# ---- 1b. _try_complete_limit: не подставляет искажённое число, а переспрашивает ----

def test_try_complete_limit_parses_thousand_unit_correctly():
    # Регрессия ровно по фидбэку: "250 тыс руб" обязано стать лимитом
    # 250000, а не 250.
    ctx = Ctx("husband", "private", 901)
    try:
        out = _try_complete_limit("250 тыс руб", ctx)
        assert out is not None and "250 000" in out
        row = _current_limit_row()
        assert row[0] == 250000
    finally:
        with db() as conn:
            conn.execute("DELETE FROM dialog_state WHERE chat_id=901")
        _restore_seed_limit()

def test_try_complete_limit_rejects_amount_below_threshold_without_writing():
    # "250" без единицы — валидное число, но меньше 1000 ₽: почти наверняка
    # человека не поняли, поэтому записывать нельзя, только переспросить.
    ctx = Ctx("husband", "private", 902)
    before = _current_limit_row()
    out = _try_complete_limit("250", ctx)
    assert out is None
    assert _current_limit_row() == before, "сумма ниже порога не должна попасть в базу"

@pytest.mark.parametrize("text", ["примерно столько же", "не знаю"])
def test_try_complete_limit_reasks_on_ambiguous_answer(text):
    ctx = Ctx("husband", "private", 903)
    before = _current_limit_row()
    out = _try_complete_limit(text, ctx)
    assert out is None
    assert _current_limit_row() == before

def test_route_message_limit_answer_with_unit_end_to_end():
    # Тот же сценарий целиком через route_message (state → _try_complete_limit),
    # как в реальном диалоге Telegram.
    ctx = Ctx("husband", "private", 904)
    set_state(904, "awaiting_limit")
    try:
        out = route_message("250 тыс руб", ctx)
        assert "250 000" in out
        assert get_state(904)["state"] == "base"
        assert _current_limit_row()[0] == 250000
    finally:
        with db() as conn:
            conn.execute("DELETE FROM dialog_state WHERE chat_id=904")
        _restore_seed_limit()


# ---- 2. Инструмент set_budget_limit ----

def test_set_budget_limit_returns_old_and_new_amounts():
    """Обе величины по-прежнему возвращаются — на них строится вопрос
    подтверждения («с 140 000 ₽ на 200 000 ₽»). Изменилось другое: запись
    теперь делает подтверждение, а не сам инструмент (см. раздел 4)."""
    try:
        r = set_budget_limit(PRIV_H, 200000)
        assert r["data"]["old_amount"] == _CANONICAL_SEED_LIMIT
        assert r["data"]["amount"] == 200000
        assert r["source_keys"] == ["family_data"]
    finally:
        with db() as conn:
            conn.execute("DELETE FROM dialog_state WHERE chat_id=%s", (PRIV_H.chat_id,))
        _restore_seed_limit()

def test_set_budget_limit_rejects_below_threshold_without_writing():
    before = _current_limit_row()
    r = set_budget_limit(PRIV_H, 250)
    assert r["data"] is None
    assert r.get("error")
    assert _current_limit_row() == before, "инструмент не должен писать в базу при отказе"

def test_set_budget_limit_is_registered_in_tool_registry():
    from budget_agent import TOOL_REGISTRY
    assert "set_budget_limit" in TOOL_REGISTRY
    try:
        r = TOOL_REGISTRY["set_budget_limit"](PRIV_H, {"amount": 210000})
        assert r["data"]["amount"] == 210000
    finally:
        with db() as conn:
            conn.execute("DELETE FROM dialog_state WHERE chat_id=%s", (PRIV_H.chat_id,))
        _restore_seed_limit()

def test_agent_stages_limit_change_via_tool(monkeypatch):
    # Логика проверяется без единого живого вызова модели: structured_call
    # подменён заранее заданным сценарием шагов (тот же приём, что и у
    # test_agent_calls_tool_then_finalizes выше).
    step = NextStep(goal_progress="начало", plan_remaining_steps=["изменить лимит"],
                    decision_summary="меняю лимит по просьбе человека",
                    call={"tool": "set_budget_limit", "amount": 220000}, task_completed=False)
    final = FinalAnswer(summary="Лимит изменён: было 140 000 ₽, стало 220 000 ₽.",
                        details=[], scenarios=[], source_keys=["family_data"])
    monkeypatch.setattr("budget_agent.structured_call", _script([step, final]))
    try:
        ans, registry, any_empty_result = run_agent("измени лимит на 220 тысяч", PRIV_H)
        assert "220 000" in ans.summary
        # Сам цикл агента лимит не пишет: инструмент готовит смену, запись
        # делает подтверждение через route_message (см. раздел 4).
        assert _current_limit_row()[0] == _CANONICAL_SEED_LIMIT
        assert get_state(PRIV_H.chat_id, PRIV_H.user_id)["state"] == "awaiting_limit_confirm"
    finally:
        with db() as conn:
            conn.execute("DELETE FROM dialog_state WHERE chat_id=%s", (PRIV_H.chat_id,))
        _restore_seed_limit()


def test_agent_executes_successful_side_effect_tool_only_once(monkeypatch):
    """Даже если модель повторяет тот же шаг, успешная запись завершается один раз."""
    from budget_agent import TOOL_REGISTRY

    step = NextStep(goal_progress="начало", plan_remaining_steps=["изменить лимит"],
                    decision_summary="меняю лимит",
                    call={"tool": "set_budget_limit", "amount": 220000},
                    task_completed=False)
    final = FinalAnswer(summary="Лимит изменён.", details=[], scenarios=[],
                        source_keys=["family_data"])
    calls = []

    def fake_structured_call(schema, messages, **kwargs):
        return final if schema is FinalAnswer else step

    def fake_set_limit(ctx, args):
        calls.append(args["amount"])
        return {"data": {"old_amount": 140000, "new_amount": args["amount"]},
                "source_keys": ["family_data"]}

    monkeypatch.setattr("budget_agent.structured_call", fake_structured_call)
    monkeypatch.setitem(TOOL_REGISTRY, "set_budget_limit", fake_set_limit)

    answer, registry, any_empty_result = run_agent(
        "измени лимит на 220 тысяч", PRIV_H, max_steps=8)

    assert answer.summary == "Лимит изменён."
    assert calls == [220000], "успешный side effect нельзя повторять"


# ---- 3. Регистрация команд Telegram в меню (set_my_commands) ----

class _FakeBotForCommands:
    def __init__(self):
        self.set_commands_calls = []
    async def set_my_commands(self, commands):
        self.set_commands_calls.append(list(commands))

class _FakeAppForPostInit:
    def __init__(self):
        self.bot = _FakeBotForCommands()

def test_post_init_registers_all_four_commands():
    app = _FakeAppForPostInit()
    asyncio.run(_post_init(app))
    assert len(app.bot.set_commands_calls) == 1, "set_my_commands должен вызываться ровно один раз"
    commands = app.bot.set_commands_calls[0]
    assert [c.command for c in commands] == ["start", "status", "report", "help"]
    assert all(c.description for c in commands), "у каждой команды должно быть описание"

def test_bot_commands_list_matches_registered_descriptions():
    assert [cmd for cmd, _ in BOT_COMMANDS] == ["start", "status", "report", "help"]
    assert all(descr for _, descr in BOT_COMMANDS)


# ===== Финальная волна правок: пункты 5 и 6 (CLI) =====

import sys
from budget_agent import main, SeedNotEmpty

def test_cli_prints_only_text_part_of_tuple(monkeypatch, capsys):
    # Пункт 6: route_message может вернуть (текст, inline_keyboard) — CLI
    # раньше печатал это как сырой Python-кортеж (`('...', {...})`).
    monkeypatch.setattr("budget_agent.route_message",
                        lambda text, ctx: ("Расходы за месяц:\nПродукты: 100 ₽",
                                            {"inline_keyboard": [[{"text": "x"}]]}))
    monkeypatch.setattr(sys, "argv", ["budget_agent.py", "/report"])
    main()
    out = capsys.readouterr().out
    assert out.strip() == "Расходы за месяц:\nПродукты: 100 ₽"
    assert "inline_keyboard" not in out

def test_cli_prints_plain_string_unchanged(monkeypatch, capsys):
    monkeypatch.setattr("budget_agent.route_message", lambda text, ctx: "просто текст")
    monkeypatch.setattr(sys, "argv", ["budget_agent.py", "какой-то вопрос"])
    main()
    assert capsys.readouterr().out.strip() == "просто текст"

def test_cli_init_reports_seed_not_empty_without_traceback(monkeypatch, capsys):
    # Пункт 5: повторный --init без --reset на уже заполненной базе должен
    # напечатать понятную фразу, а не уронить SeedNotEmpty наружу.
    monkeypatch.setattr("budget_agent.create_schema", lambda conn: None)
    monkeypatch.setattr("budget_agent.load_documents", lambda conn, path, doc_type: 0)
    def fake_load_seed(conn, force=False):
        assert force is False
        raise SeedNotEmpty("таблица transactions уже содержит 518 строк(и) — база не пуста.")
    monkeypatch.setattr("budget_agent.load_seed", fake_load_seed)
    monkeypatch.setattr(sys, "argv", ["budget_agent.py", "--init"])
    main()   # не должно поднять исключение наружу
    out = capsys.readouterr().out
    assert "SeedNotEmpty" not in out and "Traceback" not in out
    assert "--reset" in out

def test_cli_init_reset_forces_reload(monkeypatch, capsys):
    monkeypatch.setattr("budget_agent.create_schema", lambda conn: None)
    monkeypatch.setattr("budget_agent.load_documents", lambda conn, path, doc_type: 0)
    calls = []
    def fake_load_seed(conn, force=False):
        calls.append(force)
        return {"transactions": 518}
    monkeypatch.setattr("budget_agent.load_seed", fake_load_seed)
    monkeypatch.setattr(sys, "argv", ["budget_agent.py", "--init", "--reset"])
    main()
    assert calls == [True]


# ===== Финальная волна правок, повторное ревью: оговорка про видимость =====
#
# Эвристика "здесь не видно, а не не существует" в _TOOL_HEURISTICS не
# держится на живых прогонах: ревьюер прогнал сценарий 10 пять раз, из них
# 4 — плоское отрицание существования, в одном случае модель назвала
# КОНКРЕТНУЮ ЛОЖНУЮ ЦИФРУ ("сумма равна 0 рублей") вместо честного "не
# видно". Решение "нужна ли оговорка про видимость" перенесено в код
# render_answer — детерминированно по ctx + any_empty_result (третье
# значение run_agent), а не по тексту модели. Три теста ниже — прямой
# юнит-эквивалент трёх обязательных живых проверок из задания: общий чат с
# пустым результатом (да), личный чат с пустым результатом (нет — там и
# так всё видно), общий чат с непустым результатом (нет — прятать нечего).

_VISIBILITY_DISCLAIMER = ("В этом чате видны только общие данные семьи; личные записи "
                          "каждого из супругов здесь не показываются.")

def test_visibility_disclaimer_shown_in_group_when_result_empty():
    final = FinalAnswer(summary="В общих данных цель не найдена.", details=[],
                        scenarios=[], source_keys=[])
    text = render_answer(final, GROUP, "сколько я откладываю на подарок",
                         set(), True)
    assert _VISIBILITY_DISCLAIMER in text

def test_visibility_disclaimer_not_shown_in_private_chat():
    # Личный чат видит common+свой scope целиком — там нечего "не видеть от
    # самого себя", даже если конкретный инструмент вернул пустой список
    # (например, личных целей действительно нет вообще, а не только не видно).
    final = FinalAnswer(summary="Целей не найдено.", details=[],
                        scenarios=[], source_keys=[])
    text = render_answer(final, PRIV_H, "сколько я откладываю на подарок",
                         set(), True)
    assert _VISIBILITY_DISCLAIMER not in text

def test_visibility_disclaimer_not_shown_when_data_found():
    # Общий чат, но инструменты реально что-то вернули (any_empty_result
    # ложно) — прятать нечего, оговорка была бы шумом.
    final = FinalAnswer(summary="Общий бюджет: лимит 140 000 ₽, потрачено 70 000 ₽.",
                        details=[], scenarios=[], source_keys=["family_data"])
    text = render_answer(final, GROUP, "сколько мы потратили в этом месяце",
                         {"family_data"}, False)
    assert _VISIBILITY_DISCLAIMER not in text


def test_is_empty_tool_result_distinguishes_zero_from_nothing_found():
    from budget_agent import _is_empty_tool_result
    # Реальный ответ, не пустота: 0.0 или пустая валютная карта на балансе —
    # это цифра, а не "ничего не нашли".
    assert _is_empty_tool_result({"RUB": 0.0, "USD": 0.0}) is False
    assert _is_empty_tool_result({}) is False
    # Пустота — контейнер с ключом items/by_category, где ничего нет, или
    # пустой список (search_knowledge/search_household отдают data списком).
    assert _is_empty_tool_result({"items": []}) is True
    assert _is_empty_tool_result({"total": 0.0, "by_category": []}) is True
    assert _is_empty_tool_result([]) is True
    assert _is_empty_tool_result(None) is True
    # Непустой контейнер — не пустота, даже если рядом есть нулевые поля.
    assert _is_empty_tool_result({"items": [{"title": "Отпуск 2027"}]}) is False
    assert _is_empty_tool_result([{"key": "HH-001"}]) is False


def test_run_agent_any_empty_result_true_when_tool_returns_nothing(monkeypatch):
    goals_step = NextStep(goal_progress="начало", plan_remaining_steps=["цели"],
                          decision_summary="смотрю цели", call={"tool": "get_goals"},
                          task_completed=False)
    done = NextStep(goal_progress="готово", plan_remaining_steps=["ответить"],
                    decision_summary="данных нет", call={"tool": "none"}, task_completed=True)
    final = FinalAnswer(summary="Цель не найдена.", details=[], scenarios=[], source_keys=[])
    monkeypatch.setattr("budget_agent.structured_call",
                        _script([goals_step, done, final]))
    # get_goals замокан напрямую (а не через реальный сид), чтобы тест не
    # зависел от того, есть ли в сиде хоть одна ОБЩАЯ цель наравне с личной
    # (в текущем сиде есть "Отпуск 2027", common) — юнит-тест проверяет
    # именно механизм отслеживания пустоты, а не конкретные данные сида.
    monkeypatch.setattr("budget_agent.get_goals",
                        lambda ctx: {"data": {"items": []}, "source_keys": ["family_data"]})
    ans, registry, any_empty_result = run_agent("сколько я откладываю на подарок", GROUP)
    assert any_empty_result is True

def test_run_agent_any_empty_result_false_when_tool_returns_data(monkeypatch):
    balance_step = NextStep(goal_progress="начало", plan_remaining_steps=["баланс"],
                            decision_summary="смотрю баланс", call={"tool": "get_balance"},
                            task_completed=False)
    done = NextStep(goal_progress="готово", plan_remaining_steps=["ответить"],
                    decision_summary="данные есть", call={"tool": "none"}, task_completed=True)
    final = FinalAnswer(summary="Баланс есть.", details=[], scenarios=[], source_keys=["family_data"])
    monkeypatch.setattr("budget_agent.structured_call",
                        _script([balance_step, done, final]))
    ans, registry, any_empty_result = run_agent("сколько у нас денег", PRIV_H)
    assert any_empty_result is False

def test_run_agent_any_empty_result_ignores_tool_error(monkeypatch):
    # Отказ инструмента (сеть, MCP) — не сигнал о видимости: временный сбой
    # не должен включать оговорку "здесь не видно".
    cbr_step = NextStep(goal_progress="начало", plan_remaining_steps=["курс"],
                        decision_summary="смотрю курс", call={"tool": "cbr_get_rate",
                        "char_code": "USD"}, task_completed=False)
    done = NextStep(goal_progress="готово", plan_remaining_steps=["ответить"],
                    decision_summary="не получилось", call={"tool": "none"}, task_completed=True)
    final = FinalAnswer(summary="Курс получить не удалось.", details=[], scenarios=[],
                        source_keys=[])
    monkeypatch.setattr("budget_agent.structured_call",
                        _script([cbr_step, done, final]))
    monkeypatch.setattr("budget_agent.cbr_get_rate",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("MCP недоступен")))
    ans, registry, any_empty_result = run_agent("какой курс доллара", PRIV_H)
    assert any_empty_result is False


# ===== --doctor: предполётная проверка окружения =====
#
# SETTINGS вычисляется на уровне модуля через load_settings(), которая при
# нехватке обязательной переменной завершает процесс через sys.exit() — ровно
# в той ситуации, для диагностики которой --doctor и нужен. Первый тест ниже
# проверяет это не юнит-вызовом внутри уже импортированного процесса (SETTINGS
# к этому моменту уже посчитан за пределами теста), а отдельным подпроцессом.
# Копия скрипта кладётся в tmp_path без .env рядом: load_dotenv() у
# python-dotenv по умолчанию ищет .env не от os.getcwd(), а от каталога
# файла, откуда её вызвали (разбор фрейма вызова) — простой cwd=tmp_path без
# переноса самого скрипта настоящий .env рядом с budget_agent.py в репозитории
# всё равно найдёт, тест на это и наткнулся при первом прогоне.

import os
import shutil
import subprocess
import budget_agent

def test_doctor_survives_completely_missing_env(tmp_path):
    script = tmp_path / "budget_agent.py"
    shutil.copy(os.path.abspath("budget_agent.py"), script)
    # Пустое окружение: ни .env (см. комментарий выше про копию скрипта), ни
    # унаследованных переменных с именами, которые читает
    # load_settings/cmd_doctor.
    blocked_prefixes = ("PG_", "OLLAMA_", "EMBED_", "ROUTER_", "TELEGRAM_",
                         "MCP_", "LLM_", "PERSON_", "BROADCAST_", "AGENT_PLATFORM_")
    clean_env = {k: v for k, v in os.environ.items() if not k.startswith(blocked_prefixes)}
    proc = subprocess.run([sys.executable, str(script), "--doctor", "--no-llm"],
                          cwd=str(tmp_path), env=clean_env,
                          capture_output=True, text=True, timeout=30)
    assert "Traceback" not in proc.stderr, proc.stderr
    out = proc.stdout
    # Полный список проверок напечатан, включая последнюю строку — если бы
    # load_settings() отработал на уровне модуля до печати, вывода не было
    # бы вовсе (sys.exit оборвал бы процесс до main()).
    for label in ("Python 3.11+", "Зависимости", "Файл .env", "Postgres",
                  "База пустая", "Ollama", "Эмбеддинг-модель", "Языковая модель",
                  "uvx (для MCP ЦБ)", "Токен Telegram"):
        assert label in out, f"{label!r} не напечатан:\n{out}"
    # Недостающие переменные названы, а не проглочены молча.
    assert "PG_DSN" in out
    assert ".env" in out
    # Проблема есть (база недоступна как минимум) — ненулевой код возврата.
    assert proc.returncode != 0

def test_doctor_no_llm_flag_makes_zero_model_calls(monkeypatch):
    calls = {"n": 0}
    class SpyOpenAI:
        def __init__(self, *a, **kw):
            calls["n"] += 1
    monkeypatch.setattr("budget_agent.OpenAI", SpyOpenAI)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "gpt-oss:120b-cloud")
    status, detail = budget_agent._doctor_check_llm(skip=True)
    assert status == budget_agent.DOCTOR_SKIP
    assert calls["n"] == 0, "--no-llm обязан не создавать даже клиента, не то что вызывать модель"

def test_doctor_without_no_llm_does_call_the_model(monkeypatch):
    # Обратная проверка: без --no-llm проверка реально пытается вызвать
    # модель (иначе --no-llm было бы не нужно) — здесь модель замокана,
    # реального сетевого вызова нет.
    calls = {"n": 0}
    class FakeCompletions:
        @staticmethod
        def create(**kw):
            calls["n"] += 1
            class R:
                choices = [type("C", (), {"message": type("M", (), {"content": "pong"})()})]
            return R()
    class FakeChat:
        completions = FakeCompletions()
    class SpyOpenAI:
        def __init__(self, *a, **kw):
            self.chat = FakeChat()
    monkeypatch.setattr("budget_agent.OpenAI", SpyOpenAI)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "gpt-oss:120b-cloud")
    status, detail = budget_agent._doctor_check_llm(skip=False)
    assert status == budget_agent.DOCTOR_OK
    assert calls["n"] == 1

def test_doctor_llm_check_reports_429_as_quota_exhausted(monkeypatch):
    import openai, httpx as _httpx
    class FakeCompletions:
        @staticmethod
        def create(**kw):
            req = _httpx.Request("POST", "http://x/v1/chat/completions")
            resp = _httpx.Response(429, request=req)
            raise openai.RateLimitError("rate limited", response=resp, body=None)
    class FakeChat:
        completions = FakeCompletions()
    class SpyOpenAI:
        def __init__(self, *a, **kw):
            self.chat = FakeChat()
    monkeypatch.setattr("budget_agent.OpenAI", SpyOpenAI)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "gpt-oss:120b-cloud")
    status, detail = budget_agent._doctor_check_llm(skip=False)
    assert status == budget_agent.DOCTOR_PROBLEM
    assert "429" in detail and "квота" in detail

def test_doctor_reports_outdated_dialog_state_schema():
    class Result:
        def __init__(self, row): self.row = row
        def fetchone(self): return self.row
    class OldSchemaConnection:
        def execute(self, query, params=None):
            if "to_regclass" in query:
                return Result(("transactions",))
            if "information_schema.columns" in query:
                return Result(None)
            pytest.fail(f"unexpected query: {query}")
    status, detail = budget_agent._doctor_check_db_empty(OldSchemaConnection())
    assert status == budget_agent.DOCTOR_PROBLEM
    assert "dialog_state.user_id" in detail and "пересоздайте" in detail

def test_doctor_return_code_zero_when_all_ok(monkeypatch):
    ok = (budget_agent.DOCTOR_OK, "ок")
    monkeypatch.setattr("budget_agent._doctor_check_python", lambda: ok)
    monkeypatch.setattr("budget_agent._doctor_check_deps", lambda: ok)
    monkeypatch.setattr("budget_agent._doctor_check_env_file", lambda: ok)
    monkeypatch.setattr("budget_agent._doctor_check_postgres", lambda: (ok, None))
    monkeypatch.setattr("budget_agent._doctor_check_db_empty", lambda conn: ok)
    monkeypatch.setattr("budget_agent._doctor_check_ollama", lambda: ok)
    monkeypatch.setattr("budget_agent._doctor_check_embed_model", lambda: ok)
    monkeypatch.setattr("budget_agent._doctor_check_uvx", lambda: ok)
    monkeypatch.setattr("budget_agent._doctor_check_telegram_token",
                        lambda: (budget_agent.DOCTOR_WARN, "не задан"))
    assert budget_agent.cmd_doctor(no_llm=True) == 0

def test_doctor_return_code_nonzero_when_any_problem(monkeypatch):
    ok = (budget_agent.DOCTOR_OK, "ок")
    problem = (budget_agent.DOCTOR_PROBLEM, "сломано")
    monkeypatch.setattr("budget_agent._doctor_check_python", lambda: ok)
    monkeypatch.setattr("budget_agent._doctor_check_deps", lambda: ok)
    monkeypatch.setattr("budget_agent._doctor_check_env_file", lambda: ok)
    monkeypatch.setattr("budget_agent._doctor_check_postgres", lambda: (problem, None))
    monkeypatch.setattr("budget_agent._doctor_check_db_empty", lambda conn: ok)
    monkeypatch.setattr("budget_agent._doctor_check_ollama", lambda: ok)
    monkeypatch.setattr("budget_agent._doctor_check_embed_model", lambda: ok)
    monkeypatch.setattr("budget_agent._doctor_check_uvx", lambda: ok)
    monkeypatch.setattr("budget_agent._doctor_check_telegram_token", lambda: ok)
    assert budget_agent.cmd_doctor(no_llm=True) != 0

def test_cli_doctor_exits_with_returncode_from_cmd_doctor(monkeypatch):
    monkeypatch.setattr("budget_agent.cmd_doctor", lambda no_llm=False: 1)
    monkeypatch.setattr(sys, "argv", ["budget_agent.py", "--doctor", "--no-llm"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


# ===== Правка по живой обратной связи (2026-08-26) =====
#
# Очередь простоя разбиралась при старте процесса: два /start, отправленных до
# запуска бота, дали два приветствия подряд. PR #6 закрыл это флагом
# drop_pending_updates=True, но вместе с приветствиями молча терялись траты и
# ответы на висящие вопросы (замечание ревью). Теперь очередь доходит до бота,
# а не исполняется стражем _skip_stale — тест держит на месте обе половины:
# и отсутствие флага, и регистрацию стража перед остальными обработчиками.
#
# Подмена идёт по "telegram.ext.Application", а не по атрибуту budget_agent:
# run_telegram импортирует Application внутри тела функции, поэтому имя
# разрешается в момент вызова и подменённый класс подхватывается. Сеть при
# этом не задействована — до app.run_polling() настоящий Application не
# создаётся вовсе.

def test_run_polling_keeps_queue_and_registers_stale_guard(monkeypatch):
    recorded = {}
    handlers = []

    class _FakeApp:
        def add_handler(self, handler, *a, **k):
            group = k.get("group", a[0] if a else 0)
            handlers.append((handler, group))
        def run_polling(self, **kwargs):
            recorded.update(kwargs)

    class _FakeBuilder:
        def token(self, *a, **k): return self
        def request(self, *a, **k): return self
        def get_updates_request(self, *a, **k): return self
        def post_init(self, *a, **k): return self
        def build(self): return _FakeApp()

    class _FakeApplication:
        @staticmethod
        def builder(): return _FakeBuilder()

    monkeypatch.setattr("telegram.ext.Application", _FakeApplication)
    monkeypatch.setattr("budget_agent.SETTINGS", SETTINGS.__class__(
        **{**SETTINGS.__dict__, "telegram_token": "111:test-token"}))

    run_telegram()

    assert "drop_pending_updates" not in recorded, (
        "очередь не должна отбрасываться на стороне Telegram — иначе человек "
        "не узнает, что его сообщение не обработано")
    assert recorded.get("allowed_updates") == ["message", "callback_query"], (
        "подписка шире обрабатываемых типов: при Update.ALL_TYPES "
        "edited_message проходил filters.TEXT в _on_text, где resolve_ctx "
        "падал на ветке callback_query — человек, поправивший опечатку в "
        "сообщении, получал тишину и исключение в журнале")
    # Заодно фиксируем, что подмена не сломала регистрацию: страж, четыре
    # команды, свободный текст и кнопка «Совет от ИИ» — иначе тест мог бы
    # позеленеть на пустом run_telegram, ничего в действительности не проверив.
    assert len(handlers) == 7, len(handlers)
    guards = [g for h, g in handlers if g < 0]
    assert guards == [-1], "страж очереди простоя обязан идти раньше остальных обработчиков"
    assert budget_agent._STARTED_AT is not None, "момент запуска не зафиксирован"


# ===== 4. Подтверждение смены лимита (закрытие гэпа из ревью PR #4) =====
#
# Инструмент set_budget_limit писал лимит сразу. Если человек не назвал сумму,
# модель была вправе придумать число и записать его без единого вопроса: порог
# MIN_SENSIBLE_LIMIT ловит грубую ошибку разбора, но не самодеятельность.
# Теперь инструмент только готовит смену и ставит dialog_state в
# awaiting_limit_confirm, а запись делает детерминированный обработчик ответа —
# по образцу awaiting_category. Диалоговый ввод лимита (/start → awaiting_limit)
# не затронут: там число называет сам человек, подтверждать нечего.

def _cleanup_chat(chat_id: int):
    with db() as conn:
        conn.execute("DELETE FROM dialog_state WHERE chat_id=%s", (chat_id,))


def test_set_budget_limit_stages_confirmation_without_writing():
    """Инструмент агента не пишет лимит сразу — он готовит подтверждение."""
    before = _current_limit_row()
    try:
        r = set_budget_limit(PRIV_H, 210000)
        assert r["data"]["pending_confirmation"] is True
        assert r["data"]["amount"] == 210000
        assert _current_limit_row() == before, "лимит не должен меняться до подтверждения"
        st = get_state(PRIV_H.chat_id, PRIV_H.user_id)
        assert st["state"] == "awaiting_limit_confirm"
        assert st["pending"]["amount"] == 210000
    finally:
        _cleanup_chat(PRIV_H.chat_id)
        _restore_seed_limit()


def test_set_budget_limit_below_threshold_stages_nothing():
    """Отказ по порогу остаётся отказом: ни записи, ни висящего вопроса."""
    before = _current_limit_row()
    try:
        r = set_budget_limit(PRIV_H, 250)
        assert r.get("error")
        assert _current_limit_row() == before
        assert get_state(PRIV_H.chat_id, PRIV_H.user_id)["state"] == "base"
    finally:
        _cleanup_chat(PRIV_H.chat_id)


def test_agent_limit_change_asks_before_writing(monkeypatch):
    """Ответ на смену лимита — детерминированный вопрос, а не текст модели.

    Модель в сценарии рапортует «Лимит изменён» — если бы её текст уходил
    наружу как есть, человек прочитал бы о свершившейся смене, которой не было.
    """
    ctx = Ctx("husband", "private", 910)
    step = NextStep(goal_progress="начало", plan_remaining_steps=["изменить лимит"],
                    decision_summary="меняю лимит", task_completed=False,
                    call={"tool": "set_budget_limit", "amount": 220000})
    final = FinalAnswer(summary="Лимит изменён: было 140 000 ₽, стало 220 000 ₽.",
                        details=[], scenarios=[], source_keys=["family_data"])
    monkeypatch.setattr("budget_agent.structured_call", _script([step, final]))
    monkeypatch.setattr("budget_agent.parse_transaction", lambda *a, **k: None)
    try:
        out = route_message("поставь разумный месячный лимит", ctx)
        assert "220 000" in out
        assert "Лимит изменён" not in out, "текст модели не должен уходить наружу как есть"
        assert _current_limit_row()[0] == _CANONICAL_SEED_LIMIT, "запись до подтверждения"
        st = get_state(910)
        assert st["state"] == "awaiting_limit_confirm"
        assert st["pending"]["amount"] == 220000
    finally:
        _cleanup_chat(910)
        _restore_seed_limit()


def test_confirmed_limit_change_writes_new_amount():
    """«да» в состоянии awaiting_limit_confirm записывает подготовленную сумму."""
    set_state(911, "awaiting_limit_confirm",
              {"amount": 220000, "old_amount": _CANONICAL_SEED_LIMIT})
    ctx = Ctx("husband", "private", 911)
    try:
        out = route_message("да", ctx)
        assert "220 000" in out
        assert _current_limit_row()[0] == 220000
        assert get_state(911)["state"] == "base"
    finally:
        _cleanup_chat(911)
        _restore_seed_limit()


def test_declined_limit_change_keeps_old_amount():
    """«нет» снимает подготовленную смену и оставляет прежний лимит."""
    set_state(912, "awaiting_limit_confirm",
              {"amount": 220000, "old_amount": _CANONICAL_SEED_LIMIT})
    ctx = Ctx("husband", "private", 912)
    try:
        out = route_message("нет", ctx)
        assert _current_limit_row()[0] == _CANONICAL_SEED_LIMIT
        assert get_state(912)["state"] == "base"
        assert "140 000" in out
    finally:
        _cleanup_chat(912)
        _restore_seed_limit()


def test_unrelated_message_drops_prepared_limit_change(monkeypatch):
    """Человек передумал отвечать: висящая смена снимается, а не удерживается.

    Тот же принцип, что у awaiting_category и awaiting_limit — сообщение идёт
    дальше по обычным ступеням, но с уведомлением, что смена не состоялась.
    """
    set_state(913, "awaiting_limit_confirm",
              {"amount": 220000, "old_amount": _CANONICAL_SEED_LIMIT})
    ctx = Ctx("husband", "private", 913)
    monkeypatch.setattr("budget_agent.parse_transaction", lambda *a, **k: None)
    monkeypatch.setattr("budget_agent.run_agent",
                        lambda *a, **k: (FinalAnswer(summary="Вчера потрачено 3 200 ₽.",
                                                     details=[], scenarios=[],
                                                     source_keys=["family_data"]),
                                         {"family_data"}, False))
    try:
        out = route_message("сколько я потратил вчера", ctx)
        assert "3 200" in out, "исходный вопрос должен быть отвечен"
        assert "не изменён" in out, "о снятой смене лимита нужно сказать явно"
        assert _current_limit_row()[0] == _CANONICAL_SEED_LIMIT
        assert get_state(913)["state"] == "base"
    finally:
        _cleanup_chat(913)
        _restore_seed_limit()


def test_limit_texts_keep_their_punctuation():
    """Пробел вместо разделителя разрядов ставится только в числах.

    Раньше .replace(",", " ") применялся ко всей фразе целиком, и запятая
    предложения («Ответьте «да», «нет» …») пропадала вместе с разделителем
    разрядов — поймано на живом прогоне CLI.
    """
    from budget_agent import _limit_confirmation_question, _reset_notice

    q = _limit_confirmation_question({"amount": 220000, "old_amount": 140000})
    assert "220 000" in q and "140 000" in q
    assert "запишу, «нет»" in q

    notice = _reset_notice("awaiting_limit_confirm", {"amount": 220000, "old_amount": 140000})
    assert "220 000" in notice and "140 000" in notice
    assert "не изменён, прежние" in notice


def test_declined_limit_change_keeps_punctuation():
    set_state(914, "awaiting_limit_confirm",
              {"amount": 220000, "old_amount": _CANONICAL_SEED_LIMIT})
    try:
        out = route_message("нет", Ctx("husband", "private", 914))
        assert out.startswith("Хорошо, лимит не изменён")
        assert "140 000" in out
    finally:
        _cleanup_chat(914)
        _restore_seed_limit()


# ===== 5. Очередь Telegram за время простоя (follow-up к PR #6) =====
#
# PR #6 отбрасывал очередь целиком (drop_pending_updates=True). Это убирало
# двойные приветствия, но вместе с ними молча теряло реальные траты и ответы
# на висящие вопросы — замечание ревью 2026-08-26. Отменять действие поздним
# сообщением по-прежнему нельзя (удалить записанную транзакцию бот не умеет),
# поэтому устаревший ввод по-прежнему не исполняется — но перестаёт исчезать
# молча: первый обработчик отвечает один раз на чат и останавливает разбор.

import datetime as _dtest


class _FakeDatedUpdate:
    """Update с датой сообщения — её у _FakeUpdate нет, а страж смотрит именно
    на неё."""

    def __init__(self, text, date, chat_id=1, user_id=555, chat_type="private"):
        self.message = _FakeMessage(text, chat_id=chat_id, user_id=user_id,
                                     chat_type=chat_type)
        self.message.date = date
        self.callback_query = None


def _guard_started_at(monkeypatch, started_at):
    monkeypatch.setattr("budget_agent._STARTED_AT", started_at)
    monkeypatch.setattr("budget_agent._STALE_NOTIFIED", set())


def test_stale_message_is_not_processed_and_gets_a_notice(monkeypatch):
    from telegram.ext import ApplicationHandlerStop
    from budget_agent import _skip_stale

    monkeypatch.setattr("budget_agent.SETTINGS", _settings_with_persons(broadcast_steps=False))
    monkeypatch.setattr("budget_agent.route_message",
                        lambda *a, **k: pytest.fail("устаревшее сообщение не должно исполняться"))
    started = _dtest.datetime.now(_dtest.timezone.utc)
    _guard_started_at(monkeypatch, started)

    update = _FakeDatedUpdate("потратил 3200 в ВкусВилле",
                              started - _dtest.timedelta(hours=2))
    context = _FakeContext()
    with pytest.raises(ApplicationHandlerStop):
        asyncio.run(_skip_stale(update, context))
    assert len(context.bot.sent) == 1
    assert "заново" in context.bot.sent[0]["text"]


def test_second_stale_message_in_same_chat_is_silent(monkeypatch):
    from telegram.ext import ApplicationHandlerStop
    from budget_agent import _skip_stale

    monkeypatch.setattr("budget_agent.SETTINGS", _settings_with_persons(broadcast_steps=False))
    started = _dtest.datetime.now(_dtest.timezone.utc)
    _guard_started_at(monkeypatch, started)
    old = started - _dtest.timedelta(minutes=30)
    context = _FakeContext()

    for text in ("/start", "/start"):
        with pytest.raises(ApplicationHandlerStop):
            asyncio.run(_skip_stale(_FakeDatedUpdate(text, old), context))
    assert len(context.bot.sent) == 1, "предупреждение шлётся один раз на чат, а не на сообщение"


def test_fresh_message_passes_the_stale_guard(monkeypatch):
    from budget_agent import _skip_stale

    monkeypatch.setattr("budget_agent.SETTINGS", _settings_with_persons(broadcast_steps=False))
    started = _dtest.datetime.now(_dtest.timezone.utc)
    _guard_started_at(monkeypatch, started)

    update = _FakeDatedUpdate("сколько потрачено", started + _dtest.timedelta(seconds=5))
    context = _FakeContext()
    asyncio.run(_skip_stale(update, context))   # без ApplicationHandlerStop
    assert context.bot.sent == [], "свежее сообщение страж не трогает"


def test_stale_message_from_stranger_gets_no_notice(monkeypatch):
    from telegram.ext import ApplicationHandlerStop
    from budget_agent import _skip_stale

    monkeypatch.setattr("budget_agent.SETTINGS", _settings_with_persons(broadcast_steps=False))
    started = _dtest.datetime.now(_dtest.timezone.utc)
    _guard_started_at(monkeypatch, started)

    update = _FakeDatedUpdate("привет", started - _dtest.timedelta(hours=1),
                              chat_id=777, user_id=999)
    context = _FakeContext()
    with pytest.raises(ApplicationHandlerStop):
        asyncio.run(_skip_stale(update, context))
    assert context.bot.sent == [], "посторонним бот не отвечает и здесь"


# ===== 6. Доходная часть =====
#
# До этой правки доход существовал только в сид-данных: он входил в баланс
# (get_balance считает по знаку categories.kind) и в прогноз кассового разрыва,
# но спросить о нём было нечем, а сообщение «получил зарплату 250 тысяч»
# молча уходило в SGR-цикл и ничего не записывало. Живая проверка 2026-08-28:
# на вопрос «сколько мы заработали в прошлом месяце» агент ответил, что данных
# нет, хотя 228 000 ₽ лежали в таблице транзакций.

from budget_agent import (get_income, parse_transaction, ParsedTransaction, add_income,
                          CategoryGuess)


def _income_rows(chat_marker: str):
    with db() as conn:
        return conn.execute(
            "SELECT amount, merchant FROM transactions WHERE merchant = %s", (chat_marker,)
        ).fetchall()


def test_get_income_returns_total_and_categories():
    """Зеркало get_expenses по kind='income': сид даёт 228 000 ₽ в месяц."""
    r = get_income(PRIV_H, "last_month")
    assert r["source_keys"] == ["family_data"]
    assert r["data"]["total"] == 228000
    assert ("Зарплата", 228000.0) in r["data"]["by_category"]


def test_get_income_is_registered_as_tool():
    from budget_agent import TOOL_REGISTRY
    assert "get_income" in TOOL_REGISTRY
    r = TOOL_REGISTRY["get_income"](PRIV_H, {"period": "last_month"})
    assert r["data"]["total"] == 228000


def test_income_message_is_recorded_with_income_category(monkeypatch):
    """Сообщение о доходе пишется как транзакция категории kind='income'."""
    monkeypatch.setattr("budget_agent.parse_transaction",
                        lambda *a, **k: ParsedTransaction(
                            kind="income", amount=250000, counterparty="ТЕСТ_Доход_920",
                            currency="RUB"))
    ctx = Ctx("husband", "private", 920)
    try:
        out = route_message("получил зарплату 250 тысяч", ctx)
        assert "250 000" in out
        with db() as conn:
            row = conn.execute(
                """SELECT t.amount, c.kind, c.name FROM transactions t
                   JOIN categories c ON c.id = t.category_id
                   WHERE t.merchant = 'ТЕСТ_Доход_920'""").fetchone()
        assert row is not None, "доход должен попасть в таблицу транзакций"
        assert float(row[0]) == 250000
        assert row[1] == "income", "категория дохода, а не расхода"
    finally:
        with db() as conn:
            conn.execute("DELETE FROM transactions WHERE merchant='ТЕСТ_Доход_920'")
            conn.execute("DELETE FROM dialog_state WHERE chat_id=920")


def test_recorded_income_does_not_count_as_expense(monkeypatch):
    """Доход не должен попадать в расходы месяца — иначе /status соврёт."""
    monkeypatch.setattr("budget_agent.parse_transaction",
                        lambda *a, **k: ParsedTransaction(
                            kind="income", amount=99000, counterparty="ТЕСТ_Доход_921",
                            currency="RUB"))
    ctx = Ctx("husband", "private", 921)
    before = get_expenses(PRIV_H, "this_month")["data"]["total"]
    try:
        route_message("пришла премия 99 тысяч", ctx)
        after = get_expenses(PRIV_H, "this_month")["data"]["total"]
        assert after == before, "доход не расход"
    finally:
        with db() as conn:
            conn.execute("DELETE FROM transactions WHERE merchant='ТЕСТ_Доход_921'")
            conn.execute("DELETE FROM dialog_state WHERE chat_id=921")


def test_add_income_rejects_expense_category():
    """Защита от перепутанного вида: доход нельзя записать расходной категорией."""
    with pytest.raises(ValueError):
        add_income(PRIV_H, 1000, "ТЕСТ_Доход_922", category="Продукты")


def test_income_asks_when_source_is_unclear(monkeypatch):
    """Когда категорий дохода несколько и модель не уверена — бот переспрашивает,
    а ответ завершает запись."""
    with db() as conn:
        conn.execute("INSERT INTO categories (name, kind) VALUES ('ТЕСТ_Премия', 'income') "
                     "ON CONFLICT DO NOTHING")
    monkeypatch.setattr("budget_agent.parse_transaction",
                        lambda *a, **k: ParsedTransaction(
                            kind="income", amount=40000, counterparty="ТЕСТ_Источник_923",
                            currency="RUB"))
    monkeypatch.setattr("budget_agent.structured_call",
                        lambda schema, msgs, **k: CategoryGuess(category="", confident=False))
    ctx = Ctx("husband", "private", 923)
    try:
        ask = route_message("пришли деньги 40 тысяч", ctx)
        assert "ТЕСТ_Премия" in ask, "в вопросе должны быть варианты категорий дохода"
        assert get_state(923)["state"] == "awaiting_income_category"
        with db() as conn:
            assert conn.execute(
                "SELECT 1 FROM transactions WHERE merchant='ТЕСТ_Источник_923'").fetchone() is None

        done = route_message("ТЕСТ_Премия", ctx)
        assert "40 000" in done
        assert get_state(923)["state"] == "base"
        with db() as conn:
            row = conn.execute(
                """SELECT c.name FROM transactions t JOIN categories c ON c.id = t.category_id
                   WHERE t.merchant = 'ТЕСТ_Источник_923'""").fetchone()
        assert row and row[0] == "ТЕСТ_Премия"
    finally:
        with db() as conn:
            conn.execute("DELETE FROM transactions WHERE merchant='ТЕСТ_Источник_923'")
            conn.execute("DELETE FROM dialog_state WHERE chat_id=923")
            conn.execute("DELETE FROM merchant_aliases WHERE alias='тест_источник_923'")
            conn.execute("DELETE FROM categories WHERE name='ТЕСТ_Премия'")


def test_classifier_recognizes_income_on_live_model():
    """Живой разбор, без подмены: промпт обязан отличать доход от траты."""
    parsed = parse_transaction("получил зарплату 250 тысяч")
    assert parsed is not None and parsed.kind == "income"
    assert parsed.amount == 250000


def test_every_registry_tool_is_expressible_in_next_step_schema():
    """Реестр и схема шага обязаны совпадать.

    Живой случай 2026-08-28: get_income добавили в TOOL_REGISTRY и в подсказку,
    но забыли класс вызова в размеченном объединении ToolCall — модель
    физически не могла назвать инструмент и на вопрос о доходе отвечала «данных
    нет», обойдя базу. Реестр без схемы — мёртвый инструмент.
    """
    import json
    from budget_agent import TOOL_REGISTRY, NextStep

    schema = json.dumps(NextStep.model_json_schema(), ensure_ascii=False)
    missing = [name for name in TOOL_REGISTRY if f'"{name}"' not in schema]
    assert not missing, f"в реестре есть, а вызвать нельзя: {missing}"


# ===== 7. Валютные операции не исчезают из отчётов =====
#
# Ревью PR #20: доход в долларах записывался (ParsedTransaction допускает USD,
# add_income проводит его по валютному счёту), но get_income фильтрует
# t.currency = 'RUB' и возвращает один total без единого слова о том, что
# валютная операция в него не вошла. Бот подтверждал запись, а потом на вопрос
# «сколько заработали» отвечал уверенно неполной суммой. Та же дыра была и у
# расходов — она старше и досталась доходам по наследству.

def _delete_marker(marker: str):
    with db() as conn:
        conn.execute("DELETE FROM transactions WHERE merchant = %s", (marker,))


def test_usd_income_is_disclosed_by_get_income():
    """Валютный доход виден в ответе инструмента, а не молча пропадает."""
    try:
        add_income(PRIV_H, 1234, "ТЕСТ_Доход_USD", category="Зарплата", currency="USD")
        d = get_income(PRIV_H, "this_month")["data"]
        assert ("USD", 1234.0) in d["other_currencies"], \
            "валютный доход обязан быть назван отдельно от рублёвого итога"
        assert d["total"] == sum(v for _, v in d["by_category"]), "рублёвый итог считается по рублям"
    finally:
        _delete_marker("ТЕСТ_Доход_USD")


def test_usd_expense_is_disclosed_by_get_expenses():
    """Тот же контракт у расходов — дыра была общая."""
    try:
        add_expense(PRIV_H, 99, "ТЕСТ_Трата_USD", category="Продукты", currency="USD")
        d = get_expenses(PRIV_H, "this_month")["data"]
        assert ("USD", 99.0) in d["other_currencies"]
    finally:
        _delete_marker("ТЕСТ_Трата_USD")


def test_report_mentions_currency_expenses():
    """Быстрый путь /report тоже обязан признаться, а не показывать только рубли."""
    try:
        add_expense(PRIV_H, 55, "ТЕСТ_Трата_USD_2", category="Продукты", currency="USD")
        text, _ = render_report(PRIV_H)
        assert "USD" in text, "отчёт молчит о валютных тратах месяца"
    finally:
        _delete_marker("ТЕСТ_Трата_USD_2")


def test_status_mentions_currency_expenses():
    """/status сравнивает с рублёвым лимитом — про валютные траты он обязан сказать."""
    try:
        add_expense(PRIV_H, 77, "ТЕСТ_Трата_USD_3", category="Продукты", currency="USD")
        text = render_status(PRIV_H)
        assert "USD" in text
    finally:
        _delete_marker("ТЕСТ_Трата_USD_3")


def test_income_without_currency_operations_reports_empty_list():
    """Когда валютных операций нет, поле пустое — не None и не отсутствует."""
    d = get_income(PRIV_H, "last_month")["data"]
    assert d["other_currencies"] == []


def test_currency_totals_respect_category_filter():
    """Валютная часть обязана слушаться фильтра категории.

    Ревью PR #21: category применялась только к рублёвому запросу, а
    other_currencies собирал валютные траты всех категорий — вопрос «сколько
    потратили на продукты» мог вернуть доллары, потраченные на такси.
    """
    try:
        add_expense(PRIV_H, 321, "ТЕСТ_Такси_USD", category="Транспорт", currency="USD")

        d_all = get_expenses(PRIV_H, "this_month")["data"]
        assert ("USD", 321.0) in d_all["other_currencies"], \
            "без фильтра валютная трата обязана быть видна"

        d_food = get_expenses(PRIV_H, "this_month", category="Продукты")["data"]
        assert all(cur != "USD" or total != 321.0 for cur, total in d_food["other_currencies"]), \
            "валютная трата из «Транспорта» не должна попадать в ответ про «Продукты»"

        d_transport = get_expenses(PRIV_H, "this_month", category="Транспорт")["data"]
        assert ("USD", 321.0) in d_transport["other_currencies"], \
            "в своей категории валютная трата обязана остаться"
    finally:
        with db() as conn:
            conn.execute("DELETE FROM transactions WHERE merchant='ТЕСТ_Такси_USD'")
