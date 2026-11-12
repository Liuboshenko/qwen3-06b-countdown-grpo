"""
Загрузка модели и токенизатора, построение промпта и батчевая greedy-генерация.
Одни и те же функции используются при оценке базы, обученных адаптеров
и при промежуточной оценке во время обучения.
"""

import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src import config


def set_seed(seed):
    """Фиксирует seed во всех используемых генераторах случайных чисел."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, revision=config.MODEL_REVISION)
    # для батчевой генерации у декодер-модели паддинг должен быть слева
    tokenizer.padding_side = "left"
    return tokenizer


def build_user_message(numbers, target):
    """Текст задачи для модели."""
    numbers_text = ", ".join(str(number) for number in numbers)
    return config.PROMPT_TEMPLATE.format(numbers=numbers_text, target=target)


def build_prompt(tokenizer, numbers, target):
    """
    Промпт в формате чата Qwen3 с отключённым режимом рассуждений (enable_thinking=False).
    Возвращает строку, готовую к токенизации.
    """
    messages = [{"role": "user", "content": build_user_message(numbers, target)}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=config.ENABLE_THINKING,
    )
    return prompt


def load_model(adapter_path=None, dtype_name=None, device="cuda"):
    """
    Загружает базовую модель и, если указан adapter_path, применяет к ней LoRA-адаптер.
    База остаётся замороженной, обученная часть - только адаптер.
    """
    if dtype_name is None:
        dtype_name = config.MODEL_DTYPE_NAME
    dtype = getattr(torch, dtype_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        revision=config.MODEL_REVISION,
        dtype=dtype,
        attn_implementation=config.ATTN_IMPLEMENTATION,
    )
    model.to(device)

    if adapter_path is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        print("Загружен LoRA-адаптер:", adapter_path)

    model.eval()
    return model


@torch.no_grad()
def generate_greedy(model, tokenizer, prompts, batch_size=None, max_new_tokens=None):
    """
    Жадное декодирование (do_sample=False) для списка промптов.
    Возвращает список словарей: текст ответа и число сгенерированных токенов.
    """
    if batch_size is None:
        batch_size = config.EVAL_BATCH_SIZE
    if max_new_tokens is None:
        max_new_tokens = config.MAX_COMPLETION_LENGTH

    device = next(model.parameters()).device
    stop_token_ids = get_stop_token_ids(model, tokenizer)
    outputs = []

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]
        encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        encoded = encoded.to(device)

        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.pad_token_id,
        )

        # отрезаем промпт, оставляем только новые токены
        prompt_length = encoded["input_ids"].shape[1]
        new_tokens = generated[:, prompt_length:]

        for row in new_tokens:
            # длина ответа - число токенов до первого стоп-токена включительно
            # (так же длину считает GRPOTrainer)
            length = 0
            for token_id in row.tolist():
                length = length + 1
                if token_id in stop_token_ids:
                    break
            text = tokenizer.decode(row, skip_special_tokens=True)
            outputs.append({"completion": text, "completion_tokens": length})

    return outputs


def get_stop_token_ids(model, tokenizer):
    """Множество токенов, на которых заканчивается ответ: EOS из generation_config и паддинг."""
    stop_token_ids = set()
    eos_ids = model.generation_config.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    for token_id in eos_ids:
        stop_token_ids.add(token_id)
    stop_token_ids.add(tokenizer.pad_token_id)
    return stop_token_ids
