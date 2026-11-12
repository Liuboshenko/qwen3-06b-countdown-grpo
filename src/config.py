"""
Все константы проекта в одном месте: пути, seed, параметры данных, модели, LoRA и GRPO.
Значения фиксируются ДО обучения и не меняются между условиями эксперимента.
"""

import os


# ---------------------------------------------------------------------------
# Пути
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
CHECKPOINTS_DIR = os.path.join(PROJECT_DIR, "checkpoints")

TRAIN_PATH = os.path.join(DATA_DIR, "train.jsonl")
VAL_PATH = os.path.join(DATA_DIR, "val.jsonl")
HELDOUT_PATH = os.path.join(DATA_DIR, "heldout.jsonl")


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------
DATA_SEED = 42                 # генерация задач и разбиение на сплиты
TRAIN_SEEDS = [0, 1, 2]        # три запуска обучения для каждого условия


# ---------------------------------------------------------------------------
# Данные Countdown (сложность подбирается на val так, чтобы база решала 10-50%)
# ---------------------------------------------------------------------------
NUM_COUNT = 3                  # сколько исходных чисел в задаче
NUMBER_MIN = 1                 # диапазон исходных чисел
NUMBER_MAX = 99
TARGET_MIN = 1                 # диапазон цели
TARGET_MAX = 199

TRAIN_SIZE = 4000
VAL_SIZE = 200                 # только для калибровки сложности
HELDOUT_SIZE = 300             # фиксированная отложенная выборка (>= 200 по заданию)


# ---------------------------------------------------------------------------
# Модель и генерация
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"  # точный коммит на HuggingFace
ENABLE_THINKING = False        # режим без длинных рассуждений
# На GTX 1060 (Pascal) eager-внимание в 4-6 раз быстрее SDPA (замер в WORKLOG.MD)
ATTN_IMPLEMENTATION = "eager"
MAX_COMPLETION_LENGTH = 256    # максимум новых токенов в ответе (обучение и оценка)
EVAL_BATCH_SIZE = 32           # размер батча при greedy-оценке (одинаковый для всех моделей)

# ---------------------------------------------------------------------------
# LoRA (предварительно, уточняется после замеров на GPU)
# ---------------------------------------------------------------------------
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.0             # без dropout: логвероятности политики и референса считаются одинаково
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# ---------------------------------------------------------------------------
# GRPO (предварительно, уточняется после замеров на GPU)
# ---------------------------------------------------------------------------
MODEL_DTYPE_NAME = "float16"   # Pascal не поддерживает bf16
USE_FP16_AMP = True            # смешанная точность со scaler-ом градиентов
NUM_GENERATIONS = 8            # размер группы G: ответов на одну задачу
PER_DEVICE_BATCH_SIZE = 8      # ответов в одном микробатче для backward
STEPS_PER_GENERATION = 4       # генерация батчем 8 * 4 = 32 ответа (4 задачи x 8)
GRADIENT_ACCUMULATION_STEPS = 8  # 8 * 8 = 64 ответа (8 задач x 8) на один шаг оптимизатора
MAX_STEPS = 150                # выбрано по пилоту и бюджету GPU (WORKLOG.MD)
LEARNING_RATE = 2e-5
LR_SCHEDULER = "constant_with_warmup"
WARMUP_STEPS = 10
TRAIN_TEMPERATURE = 1.0        # температура сэмплирования при обучении
KL_BETA = 0.04                 # коэффициент KL к исходной модели (значение из статьи GRPO)
LOSS_TYPE = "dapo"             # агрегирование потерь по токенам (значение по умолчанию в TRL)
SCALE_REWARDS = "group"        # нормировка преимущества на std внутри группы (как в GRPO)
GRADIENT_CHECKPOINTING = True  # экономия памяти ценой повторного прямого прохода
SAVE_STEPS = 10                # чекпойнт каждые 10 шагов
EVAL_EVERY_STEPS = 25          # greedy-оценка на held-out каждые 25 шагов

PROMPT_TEMPLATE = (
    "Using the numbers {numbers}, create an arithmetic expression that equals {target}.\n"
    "Rules:\n"
    "- Use each number exactly once.\n"
    "- You may use only +, -, *, / and parentheses.\n"
    "- Do not use any other numbers or operations, and do not concatenate digits.\n"
    "Think briefly, then write the final expression inside <answer></answer> tags "
    "without \"= {target}\" at the end.\n"
    "Example: numbers 3, 7, 25 and target 46 -> <answer>25 + 3 * 7</answer>"
)
