"""
Замер возможностей GPU для планирования экспериментов (результаты записываются в WORKLOG.MD).

1. Скорость матричного умножения в fp32 и fp16 (на Pascal fp16 может быть медленным).
2. Память и скорость greedy-генерации Qwen3-0.6B при разных размерах батча.

Запуск из корня проекта:
    python -m src.benchmark_gpu --part matmul
    python -m src.benchmark_gpu --part generate --dtype float16 --batch_sizes 8 16 32
"""

import argparse
import time

import torch

from src import config
from src.countdown_data import load_jsonl
from src.model_utils import build_prompt, generate_greedy, load_model, load_tokenizer


def benchmark_matmul():
    size = 2048
    for dtype in [torch.float32, torch.float16]:
        a = torch.randn(size, size, device="cuda", dtype=dtype)
        b = torch.randn(size, size, device="cuda", dtype=dtype)
        # прогрев
        for _ in range(3):
            torch.matmul(a, b)
        torch.cuda.synchronize()

        repeats = 20
        start = time.time()
        for _ in range(repeats):
            torch.matmul(a, b)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        tflops = 2 * size ** 3 * repeats / elapsed / 1e12
        print("matmul", size, "x", size, str(dtype), "->", round(tflops, 2), "TFLOPS")


def benchmark_generate(dtype_name, batch_sizes, num_prompts, tasks_path):
    tokenizer = load_tokenizer()
    torch.cuda.reset_peak_memory_stats()
    model = load_model(dtype_name=dtype_name)
    weights_mb = torch.cuda.memory_allocated() / 2 ** 20
    print("dtype:", dtype_name, "| память под веса, МБ:", round(weights_mb))

    tasks = load_jsonl(tasks_path)[:num_prompts]
    prompts = []
    for task in tasks:
        prompts.append(build_prompt(tokenizer, task["numbers"], task["target"]))
    prompt_lengths = [len(tokenizer(p)["input_ids"]) for p in prompts]
    print("Длина промпта, токенов: мин", min(prompt_lengths), "макс", max(prompt_lengths))

    for batch_size in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        outputs = generate_greedy(model, tokenizer, prompts, batch_size=batch_size)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        peak_mb = torch.cuda.max_memory_allocated() / 2 ** 20
        lengths = [output["completion_tokens"] for output in outputs]
        truncated = sum(1 for length in lengths if length >= config.MAX_COMPLETION_LENGTH)
        print("batch", batch_size,
              "| время на", len(prompts), "задач, сек:", round(elapsed, 1),
              "| сек на задачу:", round(elapsed / len(prompts), 3),
              "| пик памяти, МБ:", round(peak_mb),
              "| средняя длина ответа:", round(sum(lengths) / len(lengths), 1),
              "| обрезано по лимиту:", truncated)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["matmul", "generate"], required=True)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--num_prompts", type=int, default=64)
    parser.add_argument("--tasks_path", default=config.VAL_PATH)
    args = parser.parse_args()

    print("GPU:", torch.cuda.get_device_name(0))
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    print("Свободно VRAM, МБ:", round(free_bytes / 2 ** 20), "из", round(total_bytes / 2 ** 20))

    if args.part == "matmul":
        benchmark_matmul()
    else:
        benchmark_generate(args.dtype, args.batch_sizes, args.num_prompts, args.tasks_path)


if __name__ == "__main__":
    main()
