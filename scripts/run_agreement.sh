#!/usr/bin/env bash
# Per-example option loglikelihoods for every council player.
# One GPU pass buys an unlimited offline (beta,tau) sweep: the equilibrium is
# solved over the K answer options, so aggregation needs no shared tokenizer
# and every comparison (equilibrium / averaging / best-single / oracle) is
# computed from the same stored scores on identical examples.
set -u
cd /home/sharaths/projects/game-llm
OUT=results/scale/agree
TASKS=arc_challenge,hellaswag,piqa,winogrande,mmlu
for M in Qwen/Qwen3-1.7B Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-Math-1.5B-Instruct Qwen/Qwen2.5-Coder-1.5B-Instruct; do
  SAFE=${M//\//_}
  echo "$(date -Is) === agreement: $M ==="
  .venv-scale/bin/python -m lm_eval --model hf \
    --model_args pretrained=$M,dtype=bfloat16,device_map=cuda:0 \
    --tasks $TASKS --limit 150 --batch_size 8 \
    --log_samples --output_path $OUT/$SAFE
  RC=$?
  echo "$(date -Is) === $M rc=$RC ==="
done
echo "AGREEMENT SWEEP COMPLETE"
