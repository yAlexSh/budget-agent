"""Прогон 12 демо-сценариев задачи 12 (см. .superpowers/sdd/2026-08-23-budget-agent/
task-12-brief.md, шаг 3 и docs/2026-08-23-design.md, раздел 12).

Вызывает route_message для каждого сценария на живом бэкенде (Ollama,
OLLAMA_MODEL=gpt-oss:120b-cloud — см. .env), печатает вопрос, ответ и
затраченное время, в конце — сводку по p95 задержки.

Требует уже развёрнутую и проиндексированную базу (--init) — сам скрипт
ничего не индексирует и не заполняет.

Private-сценарии идут в одном и том же личном чате (PRIVATE_CHAT_ID) — это
одна непрерывная переписка мужа с ботом, как в реальном Telegram; групповой
сценарий (10) — в отдельном чате (GROUP_CHAT_ID), потому что это физически
другой чат. Состояние диалога (dialog_state), если сценарий его оставит
висящим (например, каскад категоризации не уверен и переспрашивает), не
сбрасывается вручную между сценариями: route_message сама решает, как вести
себя со свободным вопросом при висящем состоянии (см. её докстринг) — это
часть того, что сценарии должны продемонстрировать как есть, без подгонки.

Раунд ревью 3 задачи 12: формулировка сценария 7 заменена. Исходный текст
(«Отпуск обойдётся в 280 тысяч — тянем?») целиком в рублях — курсу ЦБ там
объективно нечего делать, и модель была права, что его не звала; ожидание
«три источника обязаны сойтись» было неверным для ЭТОГО текста, а не
поведение агента дефектным. Новая формулировка вводит долларовую часть
накоплений по существу (используется реальный семейный факт — валютные
накопления, HH-005), поэтому конвертация по курсу ЦБ становится частью
самого расчёта, а не вызовом ради галочки. Подробности и честное различие
«исправление ожидания vs подгонка сценария» — в demo_scenarios.md, раздел
«Раунд ревью 3», и в отчёте задачи 12.
"""
import time

from budget_agent import Ctx, route_message

SCENARIOS = [
    ("husband", "private", "Сколько мы потратили на продукты в июле?"),
    ("husband", "private", "Потратил 3200 в ВкусВилле"),
    ("husband", "private", "Потратил 4100 в Ашане"),
    ("husband", "private", "Сначала гасить кредит или копить подушку?"),
    ("husband", "private", "Хватит ли нам денег до зарплаты?"),
    ("husband", "private", "Сколько у нас всего накоплений в рублях?"),
    ("husband", "private",
     "Отпуск обойдётся в 280 тысяч — можно ли по нашим правилам закрыть часть "
     "суммы из долларовых накоплений, и хватит ли этого по текущему курсу?"),
    ("husband", "private", "Сколько закладывать на отпуск в этом году?"),
    ("husband", "private", "На сколько нам хватит подушки?"),
    ("husband", "group",   "Сколько я откладываю на подарок?"),
    ("husband", "private", "/status"),
    ("husband", "private", "/report"),
]

PRIVATE_CHAT_ID = 1001
GROUP_CHAT_ID = 2002


def _chat_id(chat_type: str) -> int:
    return PRIVATE_CHAT_ID if chat_type == "private" else GROUP_CHAT_ID


def main() -> None:
    durations: list[float] = []
    for i, (person, chat_type, text) in enumerate(SCENARIOS, start=1):
        ctx = Ctx(person=person, chat_type=chat_type, chat_id=_chat_id(chat_type))
        t0 = time.monotonic()
        result = route_message(text, ctx)
        dt = time.monotonic() - t0
        durations.append(dt)
        answer, markup = result if isinstance(result, tuple) else (result, None)

        print(f"===== Сценарий {i}/12 ({person}, {chat_type}) =====")
        print(f"Вопрос: {text}")
        print(f"Ответ:\n{answer}")
        if markup:
            print(f"Разметка: {markup}")
        print(f"Время: {dt:.2f} с")
        print()

    n = len(durations)
    ranked = sorted(durations)
    p95_index = min(n - 1, round(0.95 * (n - 1)))
    p95 = ranked[p95_index]
    print("===== Сводка =====")
    print(f"Сценариев: {n}")
    print(f"Мин: {ranked[0]:.2f} с   Медиана: {ranked[n // 2]:.2f} с   "
          f"p95: {p95:.2f} с   Макс: {ranked[-1]:.2f} с")
    print(f"Суммарное время: {sum(durations):.2f} с")


if __name__ == "__main__":
    main()
