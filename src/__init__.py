"""Код проекта: RL-дообучение Qwen3-0.6B на задаче Countdown (GRPO + LoRA)."""

import os

# Уменьшает фрагментацию видеопамяти. Без этой настройки на GTX 1060 генерация батчем 64
# падала с OOM, хотя 822 МБ были зарезервированы, но не использовались (см. WORKLOG.MD).
# Должна быть установлена до первого обращения к CUDA, поэтому стоит в __init__ пакета.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
