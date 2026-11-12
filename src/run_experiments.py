"""
Очередь запусков обучения: условия x seed.

Каждый запуск выполняется отдельным процессом (python -m src.train_grpo), чтобы после
него полностью освобождалась видеопамять. Завершённые запуски (есть results/<run>/DONE.json)
пропускаются, незавершённые продолжаются с последнего чекпойнта. Поэтому после
выключения машины достаточно запустить ту же команду ещё раз.

Запуск из корня проекта (вывод каждого запуска пишется в logs/<run>.log):
    python -m src.run_experiments --conditions baseline
"""

import argparse
import os
import subprocess
import sys
import time

from src import config


def main():
    parser = argparse.ArgumentParser(description="Очередь запусков GRPO с продолжением")
    parser.add_argument("--conditions", nargs="+", default=["baseline"])
    parser.add_argument("--seeds", type=int, nargs="+", default=config.TRAIN_SEEDS)
    args = parser.parse_args()

    logs_dir = os.path.join(config.PROJECT_DIR, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    for condition in args.conditions:
        for seed in args.seeds:
            run_name = condition + "_seed" + str(seed)
            done_path = os.path.join(config.RESULTS_DIR, run_name, "DONE.json")
            if os.path.exists(done_path):
                print("[пропуск] уже завершён:", run_name, flush=True)
                continue

            log_path = os.path.join(logs_dir, run_name + ".log")
            command = [sys.executable, "-m", "src.train_grpo", "--condition", condition, "--seed", str(seed)]
            start_text = time.strftime("%Y-%m-%d %H:%M:%S")
            print("[старт]", run_name, "| лог:", log_path, "|", start_text, flush=True)

            start_time = time.time()
            with open(log_path, "a", encoding="utf-8") as log_file:
                result = subprocess.run(command, cwd=config.PROJECT_DIR, stdout=log_file, stderr=subprocess.STDOUT)
            elapsed_minutes = (time.time() - start_time) / 60

            if result.returncode != 0:
                print("[ошибка]", run_name, "код возврата", result.returncode, "- см. лог", log_path, flush=True)
                print("Очередь остановлена. После исправления запустите команду снова - запуск продолжится.", flush=True)
                return
            print("[готово]", run_name, "| минут:", round(elapsed_minutes, 1), flush=True)

    print("Все запуски очереди завершены", flush=True)


if __name__ == "__main__":
    main()
