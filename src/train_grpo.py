"""
Обучение Qwen3-0.6B на задаче Countdown методом GRPO (trl.GRPOTrainer) с LoRA-адаптером.

- База заморожена, обучается только LoRA-адаптер.
- KL считается относительно исходной модели (адаптер отключён), beta > 0.
- Каждые SAVE_STEPS шагов сохраняется чекпойнт. При повторном запуске той же команды
  обучение продолжается с последнего чекпойнта (нужно, если машину выключали).
- Логи по шагам (награда, KL, длина ответа, доли статусов проверки) дописываются в
  results/<run_name>/train_log.jsonl.
- Каждые EVAL_EVERY_STEPS шагов модель оценивается greedy-декодированием на held-out.
  Результаты пишутся в results/<run_name>/heldout_curve.jsonl. Held-out НЕ используется
  для выбора чекпойнта: итоговая модель - последний шаг.
- Для пилотных запусков (проверка, что обучение идёт) оценка делается на val: --eval_split val.
  Так held-out не участвует ни в каком выборе параметров.

Запуск из корня проекта:
    python -m src.train_grpo --condition baseline --seed 0
"""

import argparse
import json
import os
import time

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer

from src import config
from src.countdown_data import load_jsonl
from src.evaluate import evaluate_tasks, save_evaluation
from src.model_utils import build_prompt, load_model, load_tokenizer, set_seed
from src.reward import partial_closeness_reward, strict_binary_reward


# Функции награды для каждого условия эксперимента (это единственное, чем условия отличаются).
# baseline - строгая бинарная награда из постановки задания.
# partial  - 1 за верный ответ + частичная награда до 0.5 за близость к цели (при соблюдении правил).
REWARD_FUNCTIONS = {
    "baseline": strict_binary_reward,
    "partial": partial_closeness_reward,
}


def read_jsonl_steps(path):
    """Возвращает множество шагов, уже записанных в JSONL-файл (нужно при продолжении)."""
    steps = set()
    if not os.path.exists(path):
        return steps
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                steps.add(json.loads(line)["step"])
    return steps


def append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


class JsonlLoggerCallback(TrainerCallback):
    """Дописывает каждую запись лога тренера (reward, kl, длина и т.д.) в JSONL-файл."""

    def __init__(self, log_path):
        self.log_path = log_path

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        record = {"step": state.global_step, "time": time.time()}
        for key, value in logs.items():
            record[key] = value
        append_jsonl(self.log_path, record)


class PeriodicEvalCallback(TrainerCallback):
    """
    Каждые eval_every шагов оценивает текущую модель на отложенных задачах (greedy).
    Уже оценённые шаги пропускаются, поэтому при продолжении после остановки
    оценка не дублируется.
    """

    def __init__(self, tokenizer, eval_tasks, eval_split, eval_every, run_dir):
        self.tokenizer = tokenizer
        self.eval_tasks = eval_tasks
        self.eval_split = eval_split
        self.eval_every = eval_every
        self.run_dir = run_dir
        self.curve_path = os.path.join(run_dir, eval_split + "_curve.jsonl")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = state.global_step
        is_eval_step = (step % self.eval_every == 0) or (step == state.max_steps)
        if not is_eval_step:
            return
        if step in read_jsonl_steps(self.curve_path):
            print("Шаг", step, "уже оценён на", self.eval_split, "- пропускаем")
            return

        print("Оценка на", self.eval_split, "- шаг", step)
        torch.cuda.empty_cache()
        start_time = time.time()
        records, summary = evaluate_tasks(model, self.tokenizer, self.eval_tasks)
        elapsed = time.time() - start_time
        torch.cuda.empty_cache()

        summary["step"] = step
        summary["eval_seconds"] = elapsed
        append_jsonl(self.curve_path, summary)
        answers_dir = os.path.join(self.run_dir, self.eval_split + "_answers")
        save_evaluation(records, summary, answers_dir, "step" + str(step).zfill(4))
        print("Шаг", step, "| точность на", self.eval_split + ":", round(summary["accuracy"], 4),
              "| средняя длина:", round(summary["mean_completion_tokens"], 1),
              "| время оценки, сек:", round(elapsed, 1))


def build_train_dataset(tokenizer, tasks):
    """Датасет для GRPOTrainer: колонка prompt (уже в формате чата) + numbers и target для награды."""
    rows = []
    for task in tasks:
        rows.append({
            "prompt": build_prompt(tokenizer, task["numbers"], task["target"]),
            "numbers": task["numbers"],
            "target": task["target"],
            "id": task["id"],
        })
    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser(description="GRPO + LoRA на задаче Countdown")
    parser.add_argument("--condition", default="baseline", choices=list(REWARD_FUNCTIONS.keys()))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max_steps", type=int, default=config.MAX_STEPS)
    parser.add_argument("--run_name", default=None, help="по умолчанию <condition>_seed<seed>")
    parser.add_argument("--learning_rate", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--eval_split", default="heldout", choices=["heldout", "val"],
                        help="на чём оценивать по ходу обучения (val - только для пилотных запусков)")
    args = parser.parse_args()

    run_name = args.run_name
    if run_name is None:
        run_name = args.condition + "_seed" + str(args.seed)
    run_dir = os.path.join(config.RESULTS_DIR, run_name)
    checkpoint_dir = os.path.join(config.CHECKPOINTS_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    done_path = os.path.join(run_dir, "DONE.json")
    if os.path.exists(done_path):
        print("Запуск", run_name, "уже завершён, пропускаем")
        return

    print("Запуск:", run_name, "| условие:", args.condition, "| seed:", args.seed,
          "| шагов:", args.max_steps, "| lr:", args.learning_rate, "| оценка на:", args.eval_split)
    set_seed(args.seed)

    # ---------------- Данные ----------------
    tokenizer = load_tokenizer()
    train_tasks = load_jsonl(config.TRAIN_PATH)
    if args.eval_split == "heldout":
        eval_tasks = load_jsonl(config.HELDOUT_PATH)
    else:
        eval_tasks = load_jsonl(config.VAL_PATH)
    train_dataset = build_train_dataset(tokenizer, train_tasks)
    print("Задач в train:", len(train_dataset), "| для оценки (" + args.eval_split + "):", len(eval_tasks))

    # ---------------- Модель и LoRA ----------------
    model = load_model()
    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )

    # ---------------- Параметры GRPO ----------------
    training_args = GRPOConfig(
        output_dir=checkpoint_dir,
        seed=args.seed,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=config.LR_SCHEDULER,
        warmup_steps=config.WARMUP_STEPS,
        per_device_train_batch_size=config.PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        steps_per_generation=config.STEPS_PER_GENERATION,
        num_generations=config.NUM_GENERATIONS,
        max_completion_length=config.MAX_COMPLETION_LENGTH,
        temperature=config.TRAIN_TEMPERATURE,
        beta=config.KL_BETA,
        loss_type=config.LOSS_TYPE,
        scale_rewards=config.SCALE_REWARDS,
        fp16=config.USE_FP16_AMP,
        gradient_checkpointing=config.GRADIENT_CHECKPOINTING,
        logging_steps=1,
        save_strategy="steps",
        save_steps=config.SAVE_STEPS,
        save_total_limit=2,
        report_to="none",
        log_completions=False,
    )

    log_callback = JsonlLoggerCallback(os.path.join(run_dir, "train_log.jsonl"))
    eval_callback = PeriodicEvalCallback(tokenizer, eval_tasks, args.eval_split, config.EVAL_EVERY_STEPS, run_dir)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=REWARD_FUNCTIONS[args.condition],
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        callbacks=[log_callback, eval_callback],
    )
    trainer.model.print_trainable_parameters()

    # ---------------- Обучение с продолжением ----------------
    last_checkpoint = get_last_checkpoint(checkpoint_dir)
    if last_checkpoint is not None:
        print("Найден чекпойнт, продолжаем обучение с него:", last_checkpoint)
    else:
        print("Чекпойнтов нет, начинаем обучение с нуля")

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    trainer.train(resume_from_checkpoint=last_checkpoint)
    elapsed = time.time() - start_time
    peak_memory_mb = torch.cuda.max_memory_allocated() / 2 ** 20
    print("Время обучения в этой сессии, мин:", round(elapsed / 60, 1))
    print("Пик памяти GPU (PyTorch), МБ:", round(peak_memory_mb))

    # ---------------- Сохранение итогового адаптера ----------------
    final_adapter_dir = os.path.join(checkpoint_dir, "final_adapter")
    trainer.model.save_pretrained(final_adapter_dir)
    print("Итоговый LoRA-адаптер сохранён:", final_adapter_dir)

    done_info = {
        "run_name": run_name,
        "condition": args.condition,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "eval_split": args.eval_split,
        "final_adapter": os.path.relpath(final_adapter_dir, config.PROJECT_DIR),
        "train_seconds_last_session": elapsed,
        "peak_gpu_memory_mb_last_session": peak_memory_mb,
    }
    with open(done_path, "w", encoding="utf-8") as file:
        json.dump(done_info, file, ensure_ascii=False, indent=2)
    print("Запуск завершён:", run_name)


if __name__ == "__main__":
    main()
