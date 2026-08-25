"""Генератор сид-данных для агента семейного бюджета.

Входит в поставку (см. README, таблица файлов) — командой ниже проверяющий
может перегенерировать seed_data.json под свою дату запуска:

    .venv/bin/python make_seed.py

random.seed фиксирован константой — детерминированность обеспечивает он, а
не "замороженная" дата. ANCHOR_DATE берётся от реальной даты запуска
(date.today()), а не от даты постановки задачи: раньше он был захардкожен на
2026-08-23, и сид-данные обрывались 31 июля навсегда — на любой более поздней
дате запуска (а тем более из другого месяца) текущий календарный месяц
оказывался пустым, и /status, /report выглядели как незагруженная база.

Два запуска в один и тот же день дают побайтно одинаковый файл: ANCHOR_DATE
зависит только от date.today() (не от времени), а random.seed — константа.
Запуски в РАЗНЫЕ дни дают разные файлы — это ожидаемо и правильно: сид обязан
покрывать "недавнее прошлое и текущий месяц" относительно даты, на которой
его сгенерировали, а не одну и ту же жёстко зашитую дату навсегда.
"""
import json
import random
from calendar import monthrange
from datetime import date, datetime, time

random.seed(20260823)

# "Текущая дата" для сид-данных — реальная дата запуска генератора, не
# константа (см. докстринг модуля выше).
ANCHOR_DATE = date.today()


def _prev_months(anchor: date, n: int) -> list[tuple[int, int]]:
    """n месяцев (год, месяц), заканчивая месяцем перед anchor, по возрастанию."""
    months = []
    y, m = anchor.year, anchor.month
    for _ in range(n):
        m -= 1
        if m == 0:
            m = 12
            y -= 1
        months.append((y, m))
    return list(reversed(months))


MONTHS = _prev_months(ANCHOR_DATE, 8)   # 2025-12 .. 2026-07


def _iso(y: int, m: int, d: int, hour: int, minute: int) -> str:
    return datetime.combine(date(y, m, d), time(hour, minute)).isoformat()


PERSONS = [
    {"name": "Алексей", "role": "husband", "telegram_id": None},
    {"name": "Мария", "role": "wife", "telegram_id": None},
]

ACCOUNTS = [
    {"name": "Основная карта", "kind": "card", "currency": "RUB", "opening_balance": 180000, "scope": "common"},
    {"name": "Накопительный счёт", "kind": "deposit", "currency": "RUB", "opening_balance": 150000, "scope": "common"},
    {"name": "Вклад-резерв", "kind": "deposit", "currency": "RUB", "opening_balance": 400000, "scope": "common"},
    {"name": "Долларовая заначка", "kind": "cash", "currency": "USD", "opening_balance": 7000, "scope": "common"},
    {"name": "Личная карта мужа", "kind": "card", "currency": "RUB", "opening_balance": 40000, "scope": "husband"},
    {"name": "Личная карта жены", "kind": "card", "currency": "RUB", "opening_balance": 38000, "scope": "wife"},
]

CATEGORIES = [
    {"name": "Продукты", "kind": "expense"}, {"name": "Транспорт", "kind": "expense"},
    {"name": "Рестораны", "kind": "expense"}, {"name": "Коммуналка", "kind": "expense"},
    {"name": "Здоровье", "kind": "expense"}, {"name": "Одежда", "kind": "expense"},
    {"name": "Развлечения", "kind": "expense"}, {"name": "Связь", "kind": "expense"},
    {"name": "Подарки", "kind": "expense"}, {"name": "Образование", "kind": "expense"},
    {"name": "Зарплата", "kind": "income"},
]

RECURRING = [
    {"title": "Коммунальные платежи", "amount": 9800, "currency": "RUB", "day_of_month": 10,
     "category": "Коммуналка", "scope": "common", "note": "квартира"},
    {"title": "Мобильная связь на двоих", "amount": 1400, "currency": "RUB", "day_of_month": 5,
     "category": "Связь", "scope": "common", "note": None},
    {"title": "Абонемент в зал", "amount": 3900, "currency": "RUB", "day_of_month": 15,
     "category": "Развлечения", "scope": "wife", "note": "заморозка дважды в год"},
]

GOALS = [
    {"title": "Отпуск 2027", "target_amount": 280000, "currency": "RUB",
     "due_date": "2027-06-01", "saved_amount": 145000, "scope": "common"},
    {"title": "Подарок к годовщине", "target_amount": 150000, "currency": "RUB",
     "due_date": "2027-06-15", "saved_amount": 120000, "scope": "husband"},
    {"title": "Курсы и сертификация", "target_amount": 96000, "currency": "RUB",
     "due_date": "2026-12-01", "saved_amount": 56000, "scope": "wife"},
]

BUDGET_LIMITS = [{"period": "monthly", "amount": 140000, "currency": "RUB", "scope": "common"}]

MERCHANT_ALIASES = [
    {"alias": "пятёрочка", "category": "Продукты"}, {"alias": "перекрёсток", "category": "Продукты"},
    {"alias": "вкусвилл", "category": "Продукты"}, {"alias": "вв", "category": "Продукты"},
    {"alias": "яндекс.такси", "category": "Транспорт"}, {"alias": "самокат", "category": "Продукты"},
]

FAMILY_RULES = [
    {"key": "large_purchase_threshold", "value_num": 45000, "currency": "RUB",
     "unit": "руб", "document_key": "HH-001", "scope": "common"},
    {"key": "emergency_fund_target", "value_num": 550000, "currency": "RUB",
     "unit": "руб", "document_key": "HH-002", "scope": "common"},
    {"key": "mandatory_monthly_expenses", "value_num": 92500, "currency": "RUB",
     "unit": "руб", "document_key": "HH-002", "scope": "common"},
    {"key": "vacation_indexation_pct", "value_num": 13.5, "unit": "процент в год",
     "document_key": "HH-003", "scope": "common"},
    {"key": "personal_money_share_pct", "value_num": 20, "unit": "процент дохода",
     "document_key": "HH-006", "scope": "common"},
    {"key": "overspend_threshold", "value_num": 30000, "currency": "RUB",
     "unit": "руб", "document_key": "HH-009", "scope": "common"},
]

PRODUCT_MERCHANTS = ["пятёрочка", "перекрёсток", "вкусвилл", "вв", "самокат", "магнит", "ашан"]
TRANSPORT_MERCHANTS = ["яндекс.такси", "метро", "азс лукойл", "каршеринг", "автобус"]
RESTAURANT_MERCHANTS = ["кафе на углу", "суши-бар", "пиццерия", "бургерная", "ресторан у моста"]

PERSONAL_CATEGORIES = ["Одежда", "Здоровье", "Развлечения", "Подарки", "Образование"]

SALARY_HUSBAND = 132000
SALARY_WIFE = 96000


CATEGORY_RANGE = {
    "Продукты": (1500, 6000),
    "Транспорт": (300, 900),
    "Рестораны": (2000, 7000),
}
LIMIT = 140000


_COUNT_RANGE = {"Продукты": (12, 18), "Транспорт": (15, 25), "Рестораны": (4, 8)}
# Доли variable_budget по категориям — посчитаны от средних значений диапазонов
# и среднего числа покупок из брифа (продукты 15*3750=56250, транспорт
# 20*600=12000, рестораны 6*4500=27000 из среднего ~95250 суммарно).
_CATEGORY_SHARE = {"Продукты": 0.59, "Транспорт": 0.13, "Рестораны": 0.28}


def _split_variable_budget(variable_budget: int) -> dict[str, int]:
    """Делит variable_budget между тремя категориями пропорционально их
    типичной доле расходов; остаток (из-за округления) уходит в Рестораны,
    чтобы сумма трёх долей совпадала с variable_budget без потерь."""
    t_products = round(variable_budget * _CATEGORY_SHARE["Продукты"])
    t_transport = round(variable_budget * _CATEGORY_SHARE["Транспорт"])
    t_restaurants = variable_budget - t_products - t_transport
    return {"Продукты": t_products, "Транспорт": t_transport, "Рестораны": t_restaurants}


def _choose_count(target: int, category: str, count_range: tuple[int, int] | None = None) -> int:
    """Число покупок в пределах count-диапазона брифа, для которого target
    вообще достижим суммой n чисел из диапазона категории (n*lo <= target <=
    n*hi). Если для целого месяца ни одно n не подходит впритык (крайний
    случай) — берётся n, для которого target ближе всего к границе
    достижимого диапазона, остаток тогда зажимается в _bounded_composition.

    count_range — переопределение _COUNT_RANGE[category]: неполный текущий
    месяц (см. _gen_current_month_partial) масштабирует диапазон числа
    покупок долей прошедших дней — иначе на третий день месяца генератор
    честно пытался бы впихнуть туда 12-18 покупок продуктов, как в полном
    месяце."""
    lo, hi = CATEGORY_RANGE[category]
    n_lo, n_hi = count_range if count_range is not None else _COUNT_RANGE[category]
    feasible = [n for n in range(n_lo, n_hi + 1) if n * lo <= target <= n * hi]
    if feasible:
        return random.choice(feasible)
    return min(range(n_lo, n_hi + 1),
               key=lambda n: min(abs(target - n * lo), abs(target - n * hi)))


def _bounded_composition(n: int, category: str, total: int) -> list[int]:
    """n случайных целых из диапазона категории, сумма которых равна total
    (зажатому в достижимый для n диапазон [n*lo, n*hi], если target был вне
    его). Стандартный приём точного разбиения суммы на ограниченные части: на
    каждом шаге сужаем диапазон очередного элемента так, чтобы оставшимся
    элементам гарантированно хватило места добрать остаток до total, не выходя
    за границы категории. В отличие от прежнего масштабирования готового
    списка, здесь каждое число изначально рождается внутри [lo, hi] — выйти
    за диапазон категории невозможно по построению, а не по счастливой
    случайности."""
    lo, hi = CATEGORY_RANGE[category]
    total = min(max(total, n * lo), n * hi)
    amounts = []
    remaining = total
    for i in range(n):
        left = n - i - 1
        item_lo = max(lo, remaining - left * hi)
        item_hi = min(hi, remaining - left * lo)
        amt = random.randint(item_lo, item_hi)
        amounts.append(amt)
        remaining -= amt
    random.shuffle(amounts)
    return amounts


def _gen_month(y: int, m: int, month_index: int) -> list[dict]:
    txs = []
    last_day = monthrange(y, m)[1]

    # Целевая общая сумма месяца (scope=common): чётные индексы — заведомо ниже
    # лимита 140000 (с запасом 4000+), нечётные — заведомо выше, гарантированный,
    # а не случайный разброс "часть месяцев за лимитом, часть в норме".
    want_over = month_index % 2 == 1
    if want_over:
        target_common = random.randint(145000, 160000)
    else:
        target_common = random.randint(120000, 135000)

    recurring_common = 9800 + 1400   # коммуналка + связь, входят в scope=common
    variable_budget = target_common - recurring_common

    targets = _split_variable_budget(variable_budget)
    n_products = _choose_count(targets["Продукты"], "Продукты")
    n_transport = _choose_count(targets["Транспорт"], "Транспорт")
    n_restaurants = _choose_count(targets["Рестораны"], "Рестораны")
    products = _bounded_composition(n_products, "Продукты", targets["Продукты"])
    transport = _bounded_composition(n_transport, "Транспорт", targets["Транспорт"])
    restaurants = _bounded_composition(n_restaurants, "Рестораны", targets["Рестораны"])
    actual_common = recurring_common + sum(products) + sum(transport) + sum(restaurants)
    assert (actual_common > LIMIT) == want_over, (
        f"{y}-{m:02d}: итог {actual_common} оказался не на той стороне лимита "
        f"{LIMIT} (want_over={want_over}) — целевой коридор месяца несовместим "
        f"с диапазонами категорий из брифа при выбранных n"
    )

    for amt in products:
        d = random.randint(1, last_day)
        txs.append({
            "ts": _iso(y, m, d, random.randint(9, 21), random.randint(0, 59)),
            "amount": amt, "currency": "RUB", "account": "Основная карта",
            "category": "Продукты", "merchant": random.choice(PRODUCT_MERCHANTS),
            "comment": None, "scope": "common", "person": None,
        })
    for amt in transport:
        d = random.randint(1, last_day)
        txs.append({
            "ts": _iso(y, m, d, random.randint(7, 23), random.randint(0, 59)),
            "amount": amt, "currency": "RUB", "account": "Основная карта",
            "category": "Транспорт", "merchant": random.choice(TRANSPORT_MERCHANTS),
            "comment": None, "scope": "common", "person": None,
        })
    for amt in restaurants:
        d = random.randint(1, last_day)
        txs.append({
            "ts": _iso(y, m, d, random.randint(12, 22), random.randint(0, 59)),
            "amount": amt, "currency": "RUB", "account": "Основная карта",
            "category": "Рестораны", "merchant": random.choice(RESTAURANT_MERCHANTS),
            "comment": None, "scope": "common", "person": None,
        })

    # Регулярные платежи — в свой день месяца.
    txs.append({
        "ts": _iso(y, m, 10, 8, 0), "amount": 9800, "currency": "RUB",
        "account": "Основная карта", "category": "Коммуналка", "merchant": None,
        "comment": "квартира", "scope": "common", "person": None,
    })
    txs.append({
        "ts": _iso(y, m, 5, 8, 0), "amount": 1400, "currency": "RUB",
        "account": "Основная карта", "category": "Связь", "merchant": None,
        "comment": None, "scope": "common", "person": None,
    })
    txs.append({
        "ts": _iso(y, m, 15, 8, 0), "amount": 3900, "currency": "RUB",
        "account": "Личная карта жены", "category": "Развлечения", "merchant": None,
        "comment": "заморозка дважды в год", "scope": "wife", "person": "wife",
    })

    # Зарплаты: 5-го — мужа, 20-го — жены.
    txs.append({
        "ts": _iso(y, m, 5, 10, 0), "amount": SALARY_HUSBAND, "currency": "RUB",
        "account": "Основная карта", "category": "Зарплата", "merchant": None,
        "comment": None, "scope": "common", "person": "husband",
    })
    txs.append({
        "ts": _iso(y, m, 20, 10, 0), "amount": SALARY_WIFE, "currency": "RUB",
        "account": "Основная карта", "category": "Зарплата", "merchant": None,
        "comment": None, "scope": "common", "person": "wife",
    })

    # Личные траты мужа и жены — 3-6 в месяц каждому.
    for person, account in (("husband", "Личная карта мужа"), ("wife", "Личная карта жены")):
        for _ in range(random.randint(3, 6)):
            d = random.randint(1, last_day)
            cat = random.choice(PERSONAL_CATEGORIES)
            amt = random.randint(1000, 12000)
            txs.append({
                "ts": _iso(y, m, d, random.randint(9, 22), random.randint(0, 59)),
                "amount": amt, "currency": "RUB", "account": account,
                "category": cat, "merchant": None, "comment": None,
                "scope": person, "person": person,
            })

    return txs, actual_common


def _gen_current_month_partial(y: int, m: int, today_day: int) -> tuple[list[dict], int]:
    """Неполный текущий месяц: с 1-го числа по today_day включительно
    (ANCHOR_DATE.day), с пропорционально меньшей суммой — иначе на любой
    дате запуска, кроме дня постановки задачи, /status и /report показывали
    бы пустой текущий месяц (пункт 3 финальной волны правок).

    Только даты уже наступивших дней месяца — ни одна транзакция не может
    быть в будущем относительно ANCHOR_DATE. Регулярные платежи и зарплаты
    попадают в сид, только если их day_of_month уже прошёл; переменные
    траты (продукты/транспорт/рестораны) и личные траты масштабируются по
    доле прошедших дней месяца — и по целевой сумме, и по числу покупок
    (count_range), иначе на третий день месяца генератор пытался бы
    втиснуть туда полный месячный набор покупок."""
    last_day = monthrange(y, m)[1]
    frac = today_day / last_day
    txs = []

    # Целевая сумма месяца — пропорция от "нормального" (не завышенного)
    # месяца: неполный месяц не должен выглядеть как перерасход.
    target_common_full = random.randint(120000, 135000)
    target_common = round(target_common_full * frac)

    recurring_common = 0
    if today_day >= 10:
        recurring_common += 9800   # коммуналка
    if today_day >= 5:
        recurring_common += 1400   # связь
    variable_budget = max(0, target_common - recurring_common)

    targets = _split_variable_budget(variable_budget)
    scaled_counts = {}
    for cat, (n_lo, n_hi) in _COUNT_RANGE.items():
        lo = max(1, round(n_lo * frac))
        hi = max(lo, round(n_hi * frac))
        scaled_counts[cat] = (lo, hi)

    n_products = _choose_count(targets["Продукты"], "Продукты", scaled_counts["Продукты"])
    n_transport = _choose_count(targets["Транспорт"], "Транспорт", scaled_counts["Транспорт"])
    n_restaurants = _choose_count(targets["Рестораны"], "Рестораны", scaled_counts["Рестораны"])
    products = _bounded_composition(n_products, "Продукты", targets["Продукты"])
    transport = _bounded_composition(n_transport, "Транспорт", targets["Транспорт"])
    restaurants = _bounded_composition(n_restaurants, "Рестораны", targets["Рестораны"])
    actual_common = recurring_common + sum(products) + sum(transport) + sum(restaurants)

    for amt in products:
        d = random.randint(1, today_day)
        txs.append({
            "ts": _iso(y, m, d, random.randint(9, 21), random.randint(0, 59)),
            "amount": amt, "currency": "RUB", "account": "Основная карта",
            "category": "Продукты", "merchant": random.choice(PRODUCT_MERCHANTS),
            "comment": None, "scope": "common", "person": None,
        })
    for amt in transport:
        d = random.randint(1, today_day)
        txs.append({
            "ts": _iso(y, m, d, random.randint(7, 23), random.randint(0, 59)),
            "amount": amt, "currency": "RUB", "account": "Основная карта",
            "category": "Транспорт", "merchant": random.choice(TRANSPORT_MERCHANTS),
            "comment": None, "scope": "common", "person": None,
        })
    for amt in restaurants:
        d = random.randint(1, today_day)
        txs.append({
            "ts": _iso(y, m, d, random.randint(12, 22), random.randint(0, 59)),
            "amount": amt, "currency": "RUB", "account": "Основная карта",
            "category": "Рестораны", "merchant": random.choice(RESTAURANT_MERCHANTS),
            "comment": None, "scope": "common", "person": None,
        })

    # Регулярные платежи и зарплаты — только если день уже наступил.
    if today_day >= 10:
        txs.append({
            "ts": _iso(y, m, 10, 8, 0), "amount": 9800, "currency": "RUB",
            "account": "Основная карта", "category": "Коммуналка", "merchant": None,
            "comment": "квартира", "scope": "common", "person": None,
        })
    if today_day >= 5:
        txs.append({
            "ts": _iso(y, m, 5, 8, 0), "amount": 1400, "currency": "RUB",
            "account": "Основная карта", "category": "Связь", "merchant": None,
            "comment": None, "scope": "common", "person": None,
        })
        txs.append({
            "ts": _iso(y, m, 5, 10, 0), "amount": SALARY_HUSBAND, "currency": "RUB",
            "account": "Основная карта", "category": "Зарплата", "merchant": None,
            "comment": None, "scope": "common", "person": "husband",
        })
    if today_day >= 15:
        txs.append({
            "ts": _iso(y, m, 15, 8, 0), "amount": 3900, "currency": "RUB",
            "account": "Личная карта жены", "category": "Развлечения", "merchant": None,
            "comment": "заморозка дважды в год", "scope": "wife", "person": "wife",
        })
    if today_day >= 20:
        txs.append({
            "ts": _iso(y, m, 20, 10, 0), "amount": SALARY_WIFE, "currency": "RUB",
            "account": "Основная карта", "category": "Зарплата", "merchant": None,
            "comment": None, "scope": "common", "person": "wife",
        })

    # Личные траты мужа и жены — число масштабируется той же долей, что и
    # переменные категории выше (не меньше 1, чтобы месяц не выглядел
    # подозрительно пустым по личным счетам уже с первых дней).
    n_personal_lo = max(1, round(3 * frac))
    n_personal_hi = max(n_personal_lo, round(6 * frac))
    for person, account in (("husband", "Личная карта мужа"), ("wife", "Личная карта жены")):
        for _ in range(random.randint(n_personal_lo, n_personal_hi)):
            d = random.randint(1, today_day)
            cat = random.choice(PERSONAL_CATEGORIES)
            amt = random.randint(1000, 12000)
            txs.append({
                "ts": _iso(y, m, d, random.randint(9, 22), random.randint(0, 59)),
                "amount": amt, "currency": "RUB", "account": account,
                "category": cat, "merchant": None, "comment": None,
                "scope": person, "person": person,
            })

    return txs, actual_common


def build() -> dict:
    transactions = []
    monthly_totals = []
    for i, (y, m) in enumerate(MONTHS):
        txs, actual_common = _gen_month(y, m, i)
        transactions.extend(txs)
        monthly_totals.append((f"{y:04d}-{m:02d}", actual_common))

    # Неполный текущий месяц (пункт 3 финальной волны) — от 1-го числа по
    # ANCHOR_DATE включительно, с пропорционально меньшей суммой. Без этого
    # "this_month" в get_budget_status/get_expenses всегда пуст на любой
    # дате запуска, кроме дня, когда сид был сгенерирован.
    cur_y, cur_m = ANCHOR_DATE.year, ANCHOR_DATE.month
    cur_txs, cur_actual = _gen_current_month_partial(cur_y, cur_m, ANCHOR_DATE.day)
    transactions.extend(cur_txs)
    monthly_totals.append((f"{cur_y:04d}-{cur_m:02d} (по {ANCHOR_DATE.day:02d} число, неполный)",
                            cur_actual))

    transactions.sort(key=lambda t: t["ts"])

    data = {
        "persons": PERSONS,
        "accounts": ACCOUNTS,
        "categories": CATEGORIES,
        "recurring": RECURRING,
        "goals": GOALS,
        "budget_limits": BUDGET_LIMITS,
        "merchant_aliases": MERCHANT_ALIASES,
        "family_rules": FAMILY_RULES,
        "transactions": transactions,
    }
    return data, monthly_totals


if __name__ == "__main__":
    data, monthly_totals = build()
    with open("seed_data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"transactions: {len(data['transactions'])}")
    print("месячные суммы (scope=common, фактические):")
    for month, total in monthly_totals:
        mark = " > лимита 140000" if total > 140000 else ""
        print(f"  {month}: {total}{mark}")
