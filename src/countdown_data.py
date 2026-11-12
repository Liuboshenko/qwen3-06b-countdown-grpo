"""
Генерация задач Countdown.

Каждая задача заведомо решаема: сначала случайно выбираются исходные числа,
из них строится допустимое выражение (каждое число ровно один раз, операции + - * /),
и значение этого выражения берётся как цель. Построенное выражение сохраняется
как эталонное решение "solution" и дополнительно проверяется функцией check_answer.

Фильтры:
- цель - целое положительное число из заданного диапазона
- цель не совпадает ни с одним исходным числом и не равна их сумме (тривиальные цели)
- повторяющиеся задачи удаляются (задача = отсортированные числа + цель)
- train, val и held-out не пересекаются

Запуск из корня проекта:
    python -m src.countdown_data
"""

import json
import os
import random
from fractions import Fraction

from src.reward import check_answer


OPERATIONS = ["+", "-", "*", "/"]


def apply_operation(left_value, right_value, operation):
    """Применяет операцию к двум дробям. Возвращает None при делении на ноль."""
    if operation == "+":
        return left_value + right_value
    if operation == "-":
        return left_value - right_value
    if operation == "*":
        return left_value * right_value
    if right_value == 0:
        return None
    return left_value / right_value


def build_random_expression(numbers, rng):
    """
    Строит случайное выражение, в котором каждое число использовано ровно один раз.

    Алгоритм: держим список "кусков" (строка выражения + его значение).
    Пока кусков больше одного - берём два случайных куска и объединяем их
    случайной операцией в скобках. Так получается любое бинарное дерево.

    Возвращает (строка_выражения, значение) или (None, None) при делении на ноль.
    """
    pieces = []
    for number in numbers:
        pieces.append((str(number), Fraction(number)))

    while len(pieces) > 1:
        # выбираем два разных куска
        first_index = rng.randrange(len(pieces))
        first_piece = pieces.pop(first_index)
        second_index = rng.randrange(len(pieces))
        second_piece = pieces.pop(second_index)

        operation = rng.choice(OPERATIONS)
        new_value = apply_operation(first_piece[1], second_piece[1], operation)
        if new_value is None:
            return None, None

        new_text = "(" + first_piece[0] + " " + operation + " " + second_piece[0] + ")"
        pieces.append((new_text, new_value))

    final_text = pieces[0][0]
    final_value = pieces[0][1]

    # внешние скобки вокруг всего выражения не нужны
    if final_text.startswith("(") and final_text.endswith(")"):
        final_text = final_text[1:-1]
    return final_text, final_value


def is_trivial_target(numbers, target):
    """Тривиальная цель: совпадает с одним из чисел или с суммой всех чисел."""
    if target in numbers:
        return True
    if target == sum(numbers):
        return True
    return False


def generate_one_task(rng, num_count, number_min, number_max, target_min, target_max):
    """
    Пытается сгенерировать одну задачу. Возвращает словарь или None,
    если случайное выражение не прошло фильтры.
    """
    numbers = []
    for _ in range(num_count):
        numbers.append(rng.randint(number_min, number_max))

    expression, value = build_random_expression(numbers, rng)
    if expression is None:
        return None

    # цель должна быть целым числом из заданного диапазона
    if value.denominator != 1:
        return None
    target = int(value)
    if target < target_min or target > target_max:
        return None

    if is_trivial_target(numbers, target):
        return None

    task = {
        "numbers": numbers,
        "target": target,
        "solution": expression,
    }
    return task


def task_key(task):
    """Ключ задачи для поиска дублей: отсортированные числа + цель."""
    sorted_numbers = tuple(sorted(task["numbers"]))
    return (sorted_numbers, task["target"])


def generate_unique_tasks(total_count, seed, num_count, number_min, number_max,
                          target_min, target_max, max_attempts=10_000_000):
    """Генерирует total_count уникальных задач с фиксированным seed."""
    rng = random.Random(seed)
    tasks = []
    seen_keys = set()
    attempts = 0

    while len(tasks) < total_count:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError("Не удалось набрать нужное число уникальных задач")

        task = generate_one_task(rng, num_count, number_min, number_max, target_min, target_max)
        if task is None:
            continue

        key = task_key(task)
        if key in seen_keys:
            continue

        # контрольная проверка: эталонное решение должно проходить функцию награды
        check = check_answer("<answer>" + task["solution"] + "</answer>", task["numbers"], task["target"])
        if check["status"] != "correct":
            raise RuntimeError("Эталонное решение не прошло проверку: " + str(task))

        seen_keys.add(key)
        tasks.append(task)

    print("Сгенерировано задач:", len(tasks), "| попыток:", attempts)
    return tasks


def split_tasks(tasks, heldout_size, val_size, seed):
    """Перемешивает задачи и делит их на held-out, val и train без пересечений."""
    rng = random.Random(seed)
    shuffled = list(tasks)
    rng.shuffle(shuffled)

    heldout = shuffled[:heldout_size]
    val = shuffled[heldout_size:heldout_size + val_size]
    train_candidates = shuffled[heldout_size + val_size:]

    # Дополнительная строгость: убираем из train задачи с тем же набором чисел,
    # что у задач val или held-out (даже если цель другая).
    reserved_numbers = set()
    for task in heldout + val:
        reserved_numbers.add(tuple(sorted(task["numbers"])))
    train = []
    for task in train_candidates:
        if tuple(sorted(task["numbers"])) not in reserved_numbers:
            train.append(task)
    print("Убрано из train задач с набором чисел из val/held-out:", len(train_candidates) - len(train))

    # явная проверка, что выборки не пересекаются
    heldout_keys = set(task_key(task) for task in heldout)
    val_keys = set(task_key(task) for task in val)
    train_keys = set(task_key(task) for task in train)
    assert len(heldout_keys & val_keys) == 0
    assert len(heldout_keys & train_keys) == 0
    assert len(val_keys & train_keys) == 0
    print("Пересечений между train / val / held-out нет")

    return train, val, heldout


def assign_ids(tasks, prefix):
    """Добавляет каждой задаче идентификатор вида heldout_0007."""
    for index, task in enumerate(tasks):
        task["id"] = prefix + "_" + str(index).zfill(4)


def save_jsonl(tasks, path):
    with open(path, "w", encoding="utf-8") as file:
        for task in tasks:
            file.write(json.dumps(task, ensure_ascii=False) + "\n")
    print("Сохранено", len(tasks), "задач в", path)


def load_jsonl(path):
    tasks = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def main():
    from src import config

    tasks = generate_unique_tasks(
        total_count=config.TRAIN_SIZE + config.VAL_SIZE + config.HELDOUT_SIZE,
        seed=config.DATA_SEED,
        num_count=config.NUM_COUNT,
        number_min=config.NUMBER_MIN,
        number_max=config.NUMBER_MAX,
        target_min=config.TARGET_MIN,
        target_max=config.TARGET_MAX,
    )
    train, val, heldout = split_tasks(tasks, config.HELDOUT_SIZE, config.VAL_SIZE, config.DATA_SEED)

    assign_ids(train, "train")
    assign_ids(val, "val")
    assign_ids(heldout, "heldout")

    os.makedirs(config.DATA_DIR, exist_ok=True)
    save_jsonl(train, config.TRAIN_PATH)
    save_jsonl(val, config.VAL_PATH)
    save_jsonl(heldout, config.HELDOUT_PATH)

    print("Пример задачи:", heldout[0])


if __name__ == "__main__":
    main()
