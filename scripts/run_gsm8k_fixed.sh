#!/usr/bin/env bash
# GSM8K measured with the settings the task actually needs.
# F28 found strict-match at 0.000 for every model, which measured whether an
# instruction-tuned model emits the "#### " convention rather than whether it can
# do arithmetic. Two changes fix that: apply each model's chat template so the
# model is prompted the way it was tuned, and give generation enough room for a
# chain of thought instead of truncating it mid-derivation.
set -u
cd "$(dirname "$0")/.."
OUT=results/scale/gsm8k_fixed
for M in Qwen/Qwen3-1.7B Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-Math-1.5B-Instruct Qwen/Qwen2.5-Coder-1.5B-Instruct; do
  SAFE=${M//\//_}
  echo "$(date -Is) === gsm8k: $M ==="
  .venv-scale/bin/python -m lm_eval --model hf \
    --model_args pretrained=$M,dtype=bfloat16,device_map=cuda:0 \
    --tasks gsm8k --limit 200 --batch_size 4 \
    --apply_chat_template --fewshot_as_multiturn --num_fewshot 4 \
    --gen_kwargs max_gen_toks=512 \
    --log_samples --output_path $OUT/$SAFE
  echo "$(date -Is) === $M rc=$? ==="
done
echo "GSM8K FIXED SWEEP COMPLETE"
