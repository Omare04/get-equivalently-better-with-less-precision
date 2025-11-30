# Eval Results Summary (aligned with progress-report goals)

Progress report focus: stabilize LM perplexity (Pile-like proxy via wikitext), improve sentiment (IMDB/SST-2), and track QA (SQuAD EM) on 3090. Below is the full run log: what we tried, why it failed or helped, and how we landed on the improved GPT-2 LoRA.

## Baselines (references)
- `base_EleutherAI_gpt-neo-125m_metrics.json`: GPT-Neo-125M untouched; good LM sanity check but weak on sentiment prompts; serves as a reference only.
- `base_gpt2_metrics.json`: GPT-2 untouched; wikitext perplexity 94.0, IMDB 0.66, SST-2 0.53. Main comparison point for GPT-2 LoRA runs.

## Attempts that did NOT meet objectives
- `wt103_eval.json` (GPT-Neo LoRA on wikitext-103):
  - Config: large wt-103 corpus, higher LR (aggressive schedule).
  - Outcome: LM perplexity worsened; sentiment did not improve.
  - Root cause: wt-103 long-tail + aggressive LR overwhelmed a small adapter on a 125M model.
- `gpt2_eval.json` (short GPT-2 LoRA run):
  - Config: shorter schedule, higher peak LR.
  - Outcome: slight sentiment gain, LM perplexity degraded.
  - Root cause: too few steps and LR too high → overfit to prompts, lost LM quality.

## Final improved run (meets the report goals better)
- File: `gpt2_lm_long_eval.json`
- Command:
  ```
  python src/peft_hf_run.py --model-name gpt2 \
    --dataset wikitext --dataset-config wikitext-2-raw-v1 \
    --max-steps 4000 --batch-size 8 --grad-accum 2 \
    --lr 1e-4 --warmup-ratio 0.1 --weight-decay 0.01 \
    --max-length 128 \
    --output-dir model/peft_hf_adapter_gpt2_lm_long
  ```
- Why this worked:
  - Switched to a smaller corpus (wikitext-2) to avoid wt-103 instability.
  - Lower LR + weight decay + warmup to smooth updates; grad accumulation to boost effective batch size within 3090 VRAM.
  - Balances LM and sentiment, matching progress-report intent (better perplexity and sentiment).
- Metrics vs. base GPT-2 (higher is better except loss/perplexity):
  - Wikitext loss: 4.54 → 3.99
  - Wikitext perplexity: 94.0 → 54.3
  - IMDB accuracy: 0.66 → 0.75
  - SST-2 accuracy: 0.53 → 0.81
  - SQuAD EM: unchanged at 0.0 (prompt-based QA not specifically trained; noted as pending in the report).

## Plots (inspect these)
- Metric comparisons: `src/evals/results/plots/wikitext_loss.png`, `wikitext_perplexity.png`, `imdb_accuracy.png`, `sst2_accuracy.png`, `squad_exact_match.png`
- Training curve (loss over steps): `src/evals/results/plots/gpt2_lm_long_training.png`
- Training dynamics (loss, grad norm, LR): `src/evals/results/plots/gpt2_lm_long_dynamics.png`

## Regenerate plots
```
python src/evals/plot_eval_results.py \
  --inputs src/evals/results/base_gpt2_metrics.json src/evals/results/gpt2_lm_long_eval.json \
  --output-dir src/evals/results/plots
python src/evals/plot_training_curve.py \
  --trainer-state model/peft_hf_adapter_gpt2_lm_long/checkpoint-4000/trainer_state.json \
  --output src/evals/results/plots/gpt2_lm_long_training.png
python src/evals/plot_training_dynamics.py \
  --trainer-state model/peft_hf_adapter_gpt2_lm_long/checkpoint-4000/trainer_state.json \
  --output src/evals/results/plots/gpt2_lm_long_dynamics.png
```

## Hardware (constant across runs)
- GPU: NVIDIA GeForce RTX 3090 (24GB), CUDA available
- Torch: 2.5.1+cu121, Python: 3.10.18
