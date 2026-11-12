# qwen3-06b-countdown-grpo
GRPO + LoRA fine-tuning of Qwen3-0.6B on Countdown: strict 0/1 reward vs partial reward for closeness to the target. 3 seeds per condition, 300-task held-out set, greedy eval. Both reach ~59% (base 20.3%), the partial reward caused a reward hacking episode. Full code, logs, tests, adapters on Hugging Face.
