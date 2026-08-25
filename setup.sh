#!/usr/bin/env bash
# Первичная настройка «Агента семейного бюджета».
#
# Делает только механическое: создаёт .venv (если его ещё нет), ставит
# зависимости, копирует .env.example в .env (если .env ещё нет), спрашивает
# два значения, которые нельзя угадать автоматически (строку подключения к
# Postgres и токен Telegram-бота), и в конце запускает предполётную
# проверку --doctor. Всё остальное в .env.example уже разумные значения по
# умолчанию — их этот скрипт не трогает.
#
# Безопасен при повторном запуске: не пересоздаёт .venv, если он уже есть,
# не копирует .env.example поверх существующего .env, а вопросы по PG_DSN и
# токену показывают уже сохранённое значение как ответ по умолчанию — пустой
# ввод его не меняет.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

DEFAULT_PG_DSN="postgresql://budget:budget@127.0.0.1:5433/budget"

# Читает значение переменной key из .env-файла file (пусто, если нет строки
# или значение не задано).
get_env_var() {
    local key="$1" file="$2"
    [ -f "$file" ] || return 0
    grep -m1 "^${key}=" "$file" 2>/dev/null | sed "s/^${key}=//"
}

# Ставит/заменяет key=value в .env-файле file, сохраняя остальные строки и
# комментарии как есть. Экранирует /, & и | — они значимы для sed с
# разделителем |.
set_env_var() {
    local key="$1" value="$2" file="$3" escaped
    escaped=$(printf '%s' "$value" | sed -e 's/[\/&|]/\\&/g')
    if grep -q "^${key}=" "$file" 2>/dev/null; then
        sed -i "s|^${key}=.*|${key}=${escaped}|" "$file"
    else
        printf '%s=%s\n' "$key" "$value" >> "$file"
    fi
}

echo "== Агент семейного бюджета: первичная настройка =="
echo

# 1. Виртуальное окружение
if [ -x ".venv/bin/python" ]; then
    echo "[1/5] .venv уже есть — пропускаю."
else
    echo "[1/5] Создаю виртуальное окружение .venv..."
    python3 -m venv .venv
fi

# 2. Зависимости
echo "[2/5] Ставлю зависимости из requirements.txt..."
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt

# 3. .env
if [ -f .env ]; then
    echo "[3/5] .env уже существует — не трогаю."
else
    echo "[3/5] Копирую .env.example в .env..."
    cp .env.example .env
fi

# 4. Единственные два значения, которые нельзя угадать автоматически
echo "[4/5] Настройка подключения"
echo

current_pg_dsn="$(get_env_var PG_DSN .env)"
default_pg_dsn="${current_pg_dsn:-$DEFAULT_PG_DSN}"
read -rp "Строка подключения к Postgres, PG_DSN [$default_pg_dsn]: " input_pg_dsn
set_env_var PG_DSN "${input_pg_dsn:-$default_pg_dsn}" .env

echo
echo "Токен Telegram-бота выдаёт @BotFather (команда /newbot в Telegram, см. README)."
echo "Без токена Telegram-слой не запустится, но режим командной строки"
echo "(budget_agent.py \"вопрос\") работает и без него."
current_tg_token="$(get_env_var TELEGRAM_BOT_TOKEN .env)"
if [ -n "$current_tg_token" ]; then
    tg_prompt_default="уже задан, Enter — оставить как есть"
else
    tg_prompt_default="Enter — пропустить, настроить позже в .env"
fi
read -rp "Токен Telegram-бота, TELEGRAM_BOT_TOKEN [$tg_prompt_default]: " input_tg_token
if [ -n "$input_tg_token" ]; then
    set_env_var TELEGRAM_BOT_TOKEN "$input_tg_token" .env
fi

# 5. Предполётная проверка
echo
echo "[5/5] Проверка окружения (--doctor)"
echo
doctor_status=0
.venv/bin/python budget_agent.py --doctor || doctor_status=$?

echo
echo "== Что дальше =="
if [ "$doctor_status" -eq 0 ]; then
    echo "Проверка пройдена без проблем. Дальше:"
else
    echo "--doctor нашёл проблемы (см. вывод выше) — устраните их перед --init,"
    echo "иначе о них станет известно только после восьми минут индексации."
    echo "После исправления перезапустите проверку: .venv/bin/python budget_agent.py --doctor"
    echo
    echo "Когда всё в порядке:"
fi
echo "  1. Индексация (~8 минут): .venv/bin/python budget_agent.py --init"
echo "  2. Проверка через CLI: .venv/bin/python budget_agent.py \"вопрос\""
echo "  3. Telegram (если задан токен): .venv/bin/python budget_agent.py --telegram"
echo "  4. Подробности и остальные шаги — README.md."

exit "$doctor_status"
