"""
Проверка ответа и функция награды для задачи Countdown.

Проверка выполняется строго по постановке задания:
1. В ответе есть выражение в тегах <answer></answer>. Если пар тегов несколько,
   проверяется последняя.
2. Выражение состоит только из целых чисел, скобок и четырёх операций + - * /.
3. Мультимножество чисел в выражении совпадает с исходным: каждое вхождение
   использовано ровно один раз.
4. Значение выражения точно равно цели.

Текст ответа модели НЕ выполняется через eval. Выражение разбирается модулем ast,
разрешены только допустимые узлы синтаксического дерева, а значение считается
вручную в точной арифметике fractions.Fraction. Деление на ноль - неверный ответ.

Базовая награда: 1.0 за верный ответ, 0.0 во всех остальных случаях.
"""

import ast
import re
from collections import Counter
from fractions import Fraction


# Ищем все пары тегов. re.DOTALL нужен, чтобы содержимое могло занимать несколько строк.
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

# Внутри тега допускаются только цифры, пробелы, скобки и четыре операции.
# Всё остальное (буквы, точка, знак равенства, ^, % и т.д.) сразу считается нарушением.
ALLOWED_CHARS_PATTERN = re.compile(r"^[0-9\s()+\-*/]+$")

# В условии задачи операции записаны знаками "×", "÷" и "−".
# Если модель использует эти символы, заменяем их на обычные * / -.
# Это те же самые четыре операции, поэтому лазейки здесь нет.
UNICODE_OPERATORS = {
    "×": "*",  # знак умножения
    "÷": "/",  # знак деления
    "−": "-",  # математический минус
}

# Статусы проверки. Нужны для анализа ошибок модели, на награду влияет только CORRECT.
STATUS_CORRECT = "correct"
STATUS_NO_ANSWER = "no_answer"              # нет пары тегов <answer></answer>
STATUS_FORBIDDEN = "forbidden_syntax"       # посторонние символы или недопустимые узлы дерева
STATUS_PARSE_ERROR = "parse_error"          # выражение не разбирается (например "25 + 3 * 7 = 46")
STATUS_WRONG_NUMBERS = "wrong_numbers"      # числа не совпадают с исходными
STATUS_DIV_ZERO = "division_by_zero"        # деление на ноль
STATUS_WRONG_VALUE = "wrong_value"          # числа верные, но значение не равно цели

ALL_STATUSES = [
    STATUS_CORRECT,
    STATUS_NO_ANSWER,
    STATUS_FORBIDDEN,
    STATUS_PARSE_ERROR,
    STATUS_WRONG_NUMBERS,
    STATUS_DIV_ZERO,
    STATUS_WRONG_VALUE,
]


class ForbiddenNodeError(Exception):
    """В выражении встретился недопустимый узел синтаксического дерева."""


def extract_answer(text):
    """Возвращает содержимое последней пары тегов <answer></answer> или None."""
    matches = ANSWER_PATTERN.findall(text)
    if len(matches) == 0:
        return None
    last_answer = matches[-1]
    return last_answer.strip()


def normalize_operators(expression):
    """Заменяет символы операций из условия (×, ÷, −) на * / -."""
    for unicode_symbol, ascii_symbol in UNICODE_OPERATORS.items():
        expression = expression.replace(unicode_symbol, ascii_symbol)
    return expression


def collect_numbers(node, numbers_found):
    """
    Рекурсивно обходит дерево, проверяет, что все узлы допустимы,
    и складывает найденные числа в список numbers_found.
    """
    if isinstance(node, ast.Constant):
        # bool в Python - подкласс int, поэтому проверяем тип строго
        if type(node.value) is not int:
            raise ForbiddenNodeError("Допускаются только целые числа")
        numbers_found.append(node.value)
        return

    if isinstance(node, ast.BinOp):
        allowed_operations = (ast.Add, ast.Sub, ast.Mult, ast.Div)
        if not isinstance(node.op, allowed_operations):
            raise ForbiddenNodeError("Недопустимая операция")
        collect_numbers(node.left, numbers_found)
        collect_numbers(node.right, numbers_found)
        return

    # Унарный минус, имена, вызовы функций, сравнения и т.д. запрещены
    raise ForbiddenNodeError("Недопустимый узел: " + type(node).__name__)


def compute_value(node):
    """Вычисляет значение уже проверенного дерева в точной арифметике Fraction."""
    if isinstance(node, ast.Constant):
        return Fraction(node.value)

    left_value = compute_value(node.left)
    right_value = compute_value(node.right)

    if isinstance(node.op, ast.Add):
        return left_value + right_value
    if isinstance(node.op, ast.Sub):
        return left_value - right_value
    if isinstance(node.op, ast.Mult):
        return left_value * right_value
    # Остался только ast.Div. Деление на ноль вызывает ZeroDivisionError.
    if right_value == 0:
        raise ZeroDivisionError("Деление на ноль")
    return left_value / right_value


def check_answer(completion, numbers, target):
    """
    Полная проверка одного ответа модели.

    Возвращает словарь:
        status     - один из статусов STATUS_*
        reward     - 1.0 если ответ верный, иначе 0.0
        expression - выражение из последнего тега (или None)
        value      - вычисленное значение в виде строки (или None)
    """
    result = {
        "status": STATUS_NO_ANSWER,
        "reward": 0.0,
        "expression": None,
        "value": None,
    }

    # Шаг 1. Достаём выражение из последней пары тегов
    expression = extract_answer(completion)
    if expression is None:
        return result
    expression = normalize_operators(expression)
    result["expression"] = expression

    # Шаг 2. Разрешены только цифры, пробелы, скобки и + - * /
    if not ALLOWED_CHARS_PATTERN.match(expression):
        result["status"] = STATUS_FORBIDDEN
        return result

    # Шаг 3. Разбор выражения модулем ast (без выполнения кода)
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        result["status"] = STATUS_PARSE_ERROR
        return result

    # Шаг 4. Проверяем узлы дерева и собираем использованные числа
    numbers_found = []
    try:
        collect_numbers(tree.body, numbers_found)
    except ForbiddenNodeError:
        result["status"] = STATUS_FORBIDDEN
        return result
    except RecursionError:
        result["status"] = STATUS_PARSE_ERROR
        return result

    # Шаг 5. Мультимножество чисел должно совпасть с исходным
    if Counter(numbers_found) != Counter(numbers):
        result["status"] = STATUS_WRONG_NUMBERS
        return result

    # Шаг 6. Точное вычисление значения
    try:
        value = compute_value(tree.body)
    except ZeroDivisionError:
        result["status"] = STATUS_DIV_ZERO
        return result
    result["value"] = str(value)

    # Шаг 7. Сравнение с целью
    if value != Fraction(target):
        result["status"] = STATUS_WRONG_VALUE
        return result

    result["status"] = STATUS_CORRECT
    result["reward"] = 1.0
    return result


def strict_binary_reward(completions, numbers, target, log_metric=None, **kwargs):
    """
    Функция награды в формате trl.GRPOTrainer (базовое условие эксперимента).

    completions - список текстов ответов модели
    numbers, target - колонки датасета, которые GRPOTrainer передаёт в kwargs
    log_metric - функция логирования от GRPOTrainer. Через неё пишем долю ответов
                 каждого статуса на шаге (нужно для анализа ошибок и поиска reward hacking).
    Возвращает список наград: 1.0 за верный ответ, 0.0 иначе.
    """
    rewards = []
    status_counts = Counter()
    for completion, task_numbers, task_target in zip(completions, numbers, target):
        check = check_answer(completion, task_numbers, task_target)
        rewards.append(check["reward"])
        status_counts[check["status"]] += 1

    if log_metric is not None:
        for status in ALL_STATUSES:
            log_metric("status/" + status, status_counts[status] / len(completions))

    return rewards


# ---------------------------------------------------------------------------
# Экспериментальное условие: частичная награда за близость к цели
# ---------------------------------------------------------------------------
# Единственное отличие от базового условия - функция награды:
#   верный ответ                                   -> 1.0 (как в baseline)
#   выражение по всем правилам, но значение неверное -> PARTIAL_MAX * близость к цели
#   любое нарушение правил (нет тега, не те числа,
#   запрещённый синтаксис, деление на ноль)         -> 0.0
# Близость = max(0, 1 - |значение - цель| / цель): 1 при точном попадании,
# 0 при отклонении на величину цели и больше.

PARTIAL_MAX = 0.5


def closeness_to_target(value, target):
    """Близость значения выражения к цели от 0 до 1 (value - Fraction, target - целое >= 1)."""
    target_fraction = Fraction(target)
    relative_error = abs(value - target_fraction) / target_fraction
    closeness = 1 - relative_error
    if closeness < 0:
        return 0.0
    return float(closeness)


def partial_reward_from_check(check, target):
    """Частичная награда по результату строгой проверки check_answer."""
    if check["status"] == STATUS_CORRECT:
        return 1.0
    if check["status"] == STATUS_WRONG_VALUE:
        value = Fraction(check["value"])
        return PARTIAL_MAX * closeness_to_target(value, target)
    return 0.0


def partial_closeness_reward(completions, numbers, target, log_metric=None, **kwargs):
    """
    Функция награды экспериментального условия в формате trl.GRPOTrainer.
    Статусы логируются так же, как в базовом условии: status/correct - это строгая
    точность на обучающих ответах, её можно сравнивать между условиями напрямую.
    """
    rewards = []
    status_counts = Counter()
    for completion, task_numbers, task_target in zip(completions, numbers, target):
        check = check_answer(completion, task_numbers, task_target)
        rewards.append(partial_reward_from_check(check, task_target))
        status_counts[check["status"]] += 1

    if log_metric is not None:
        for status in ALL_STATUSES:
            log_metric("status/" + status, status_counts[status] / len(completions))

    return rewards
