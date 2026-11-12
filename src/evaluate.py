"""
Оценка модели на наборе задач Countdown жадным декодированием.

Одна и та же процедура используется для базовой модели и для всех обученных адаптеров:
одинаковый промпт, greedy-декодирование, одинаковый max_new_tokens и размер батча,
одинаковая строгая проверка ответа.

Примеры запуска из корня проекта:
    python -m src.evaluate --split heldout --name base
    python -m src.evaluate --split heldout --name baseline_seed0 --adapter checkpoints/baseline_seed0/final_adapter
"""

import argparse
import json
import os
import time
from collections import Counter

from src import config
from src.countdown_data import load_jsonl
from src.model_utils import build_prompt, generate_greedy, load_model, load_tokenizer
from src.reward import ALL_STATUSES, check_answer


def evaluate_tasks(model, tokenizer, tasks, batch_size=None, max_new_tokens=None):
    """
    Прогоняет модель по задачам и проверяет каждый ответ.
    Возвращает (records, summary):
        records - список с ответом и результатом проверки для каждой задачи
        summary - точность, средняя длина ответа и распределение статусов
    """
    prompts = []
    for task in tasks:
        prompts.append(build_prompt(tokenizer, task["numbers"], task["target"]))

    was_training = model.training
    model.eval()
    generations = generate_greedy(model, tokenizer, prompts, batch_size, max_new_tokens)
    if was_training:
        model.train()

    records = []
    for task, generation in zip(tasks, generations):
        check = check_answer(generation["completion"], task["numbers"], task["target"])
        record = {
            "id": task["id"],
            "numbers": task["numbers"],
            "target": task["target"],
            "completion": generation["completion"],
            "completion_tokens": generation["completion_tokens"],
            "expression": check["expression"],
            "value": check["value"],
            "status": check["status"],
            "reward": check["reward"],
        }
        records.append(record)

    summary = summarize_records(records)
    return records, summary


def summarize_records(records):
    """Сводка по результатам оценки."""
    total = len(records)
    correct = 0
    total_tokens = 0
    status_counter = Counter()
    for record in records:
        correct += int(record["reward"] == 1.0)
        total_tokens += record["completion_tokens"]
        status_counter[record["status"]] += 1

    status_counts = {}
    for status in ALL_STATUSES:
        status_counts[status] = status_counter.get(status, 0)

    summary = {
        "num_tasks": total,
        "accuracy": correct / total,
        "num_correct": correct,
        "mean_completion_tokens": total_tokens / total,
        "status_counts": status_counts,
    }
    return summary


def save_evaluation(records, summary, output_dir, name):
    """Сохраняет ответы (JSONL) и сводку (JSON)."""
    os.makedirs(output_dir, exist_ok=True)
    records_path = os.path.join(output_dir, name + "_answers.jsonl")
    summary_path = os.path.join(output_dir, name + "_summary.json")

    with open(records_path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("Ответы сохранены:", records_path)
    print("Сводка сохранена:", summary_path)


def main():
    parser = argparse.ArgumentParser(description="Greedy-оценка модели на задачах Countdown")
    parser.add_argument("--split", default="heldout", choices=["heldout", "val", "train"])
    parser.add_argument("--name", required=True, help="имя модели в файлах результатов")
    parser.add_argument("--adapter", default=None, help="путь к LoRA-адаптеру (без него - базовая модель)")
    parser.add_argument("--limit", type=int, default=None, help="оценить только первые N задач")
    parser.add_argument("--output_dir", default=os.path.join(config.RESULTS_DIR, "eval"))
    args = parser.parse_args()

    split_paths = {"heldout": config.HELDOUT_PATH, "val": config.VAL_PATH, "train": config.TRAIN_PATH}
    tasks = load_jsonl(split_paths[args.split])
    if args.limit is not None:
        tasks = tasks[:args.limit]
    print("Задач для оценки:", len(tasks), "| сплит:", args.split)

    tokenizer = load_tokenizer()
    model = load_model(adapter_path=args.adapter)

    start_time = time.time()
    records, summary = evaluate_tasks(model, tokenizer, tasks)
    elapsed = time.time() - start_time

    print("Время оценки, сек:", round(elapsed, 1))
    print("Точность:", round(summary["accuracy"], 4), "(", summary["num_correct"], "из", summary["num_tasks"], ")")
    print("Средняя длина ответа, токенов:", round(summary["mean_completion_tokens"], 1))
    print("Статусы:", summary["status_counts"])

    save_evaluation(records, summary, args.output_dir, args.split + "_" + args.name)


if __name__ == "__main__":
    main()
