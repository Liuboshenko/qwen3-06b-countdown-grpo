"""
Калибровка сложности задач Countdown под базовую модель.

По заданию базовая модель должна решать примерно 10-50% задач. Для каждого варианта
сложности задачи генерируются той же процедурой и с тем же DATA_SEED, что и финальные данные,
а базовая модель оценивается ТОЛЬКО на val-сплите. Held-out при калибровке не используется.

Запуск из корня проекта:
    python -m src.calibrate_difficulty
Результаты: results/calibration/
"""

import json
import os
import time

from src import config
from src.countdown_data import assign_ids, generate_unique_tasks, split_tasks
from src.evaluate import evaluate_tasks, save_evaluation
from src.model_utils import load_model, load_tokenizer


# Варианты сложности: количество чисел, диапазон чисел, диапазон цели
CANDIDATES = [
    {"name": "n3_num1-50_t1-100", "num_count": 3, "number_min": 1, "number_max": 50, "target_min": 1, "target_max": 100},
    {"name": "n3_num1-99_t1-199", "num_count": 3, "number_min": 1, "number_max": 99, "target_min": 1, "target_max": 199},
    {"name": "n4_num1-50_t1-100", "num_count": 4, "number_min": 1, "number_max": 50, "target_min": 1, "target_max": 100},
    {"name": "n4_num1-99_t1-199", "num_count": 4, "number_min": 1, "number_max": 99, "target_min": 1, "target_max": 199},
]


def main():
    output_dir = os.path.join(config.RESULTS_DIR, "calibration")
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = load_tokenizer()
    model = load_model()

    rows = []
    for candidate in CANDIDATES:
        print("=" * 60)
        print("Вариант сложности:", candidate["name"])
        tasks = generate_unique_tasks(
            total_count=config.TRAIN_SIZE + config.VAL_SIZE + config.HELDOUT_SIZE,
            seed=config.DATA_SEED,
            num_count=candidate["num_count"],
            number_min=candidate["number_min"],
            number_max=candidate["number_max"],
            target_min=candidate["target_min"],
            target_max=candidate["target_max"],
        )
        train, val, heldout = split_tasks(tasks, config.HELDOUT_SIZE, config.VAL_SIZE, config.DATA_SEED)
        assign_ids(val, "val")

        start_time = time.time()
        records, summary = evaluate_tasks(model, tokenizer, val)
        elapsed = time.time() - start_time

        save_evaluation(records, summary, output_dir, candidate["name"])
        row = dict(candidate)
        row["val_accuracy"] = summary["accuracy"]
        row["mean_completion_tokens"] = summary["mean_completion_tokens"]
        row["status_counts"] = summary["status_counts"]
        row["eval_seconds"] = elapsed
        rows.append(row)
        print("Точность базы на val:", round(summary["accuracy"], 3),
              "| средняя длина:", round(summary["mean_completion_tokens"], 1),
              "| время, сек:", round(elapsed, 1))
        print("Статусы:", summary["status_counts"])

    with open(os.path.join(output_dir, "calibration_summary.json"), "w", encoding="utf-8") as file:
        json.dump(rows, file, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("Итог калибровки (точность базовой модели на val):")
    for row in rows:
        print(row["name"], "->", round(row["val_accuracy"], 3))


if __name__ == "__main__":
    main()
