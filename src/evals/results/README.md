# Eval Results Summary (aligned with progress-report goals)

Progress report focus: stabilize LM perplexity (Pile-like proxy via wikitext), improve sentiment (IMDB/SST-2), and track QA (SQuAD EM) on 3090. Below is the full run log: what we tried, why it failed or helped, and how we landed on the improved GPT-2 LoRA.

## Latest quick evals (adds SQuAD F1)
- Eval settings: wikitext=64, imdb=32, sst2=32, squad=16; ran on CPU (torch 2.9.1+cpu). Numbers are for quick comparisons, not full benchmarks.
- `base_gpt2_metrics.json`: ppl 54.26 (loss 3.99), IMDB 0.69, SST-2 0.44, SQuAD EM/F1 0.00 / 0.130.
- `gpt2_eval.json` (short LoRA, r=8): ppl 31.90 (loss 3.46), IMDB 0.59, SST-2 0.59, SQuAD EM/F1 0.00 / 0.102.
- `gpt2_lm_long_eval.json` (longer LoRA, r=8): ppl 31.64 (loss 3.45), IMDB 0.59, SST-2 0.56, SQuAD EM/F1 0.00 / 0.121.
- New plot: `src/evals/results/plots/squad_f1.png` alongside the existing metrics.

## Baselines (references)
- `base_EleutherAI_gpt-neo-125m_metrics.json`: GPT-Neo-125M untouched; good LM sanity check but weak on sentiment prompts; serves as a reference only.
- `base_gpt2_metrics.json`: GPT-2 untouched; quick eval baseline above (ppl 54.26, IMDB 0.69, SST-2 0.44, SQuAD F1 0.130). Main comparison point for GPT-2 LoRA runs.

## Attempts that did NOT meet objectives
- `wt103_eval.json` (GPT-Neo LoRA on wikitext-103):
  - Config: large wt-103 corpus, higher LR (aggressive schedule).
  - Outcome: LM perplexity worsened; sentiment did not improve.
  - Root cause: wt-103 long-tail + aggressive LR overwhelmed a small adapter on a 125M model.
- `gpt2_eval.json` (short GPT-2 LoRA run):
  - Config: shorter schedule, higher peak LR.
  - Outcome (quick eval): better perplexity and SST-2 than base, but IMDB dropped and SQuAD F1 stayed low.
  - Root cause: too few steps and LR too high → unstable gains; needs longer/steadier training for balanced sentiment + QA.

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
  - Balances LM quality while keeping sentiment reasonable on the constrained schedule.
- Metrics vs. base GPT-2 from the quick CPU eval (higher is better except loss/perplexity):
  - Wikitext loss: 3.99 → 3.45
  - Wikitext perplexity: 54.26 → 31.64
  - IMDB accuracy: 0.69 → 0.59
  - SST-2 accuracy: 0.44 → 0.56
  - SQuAD EM/F1: 0.00 / 0.130 → 0.00 / 0.121 (QA still untrained; EM stays at 0.0)
  - LoRA rank used: r=8 (matches the short run for apples-to-apples comparison).

## Plots (inspect these)
- Metric comparisons: `src/evals/results/plots/wikitext_loss.png`, `wikitext_perplexity.png`, `imdb_accuracy.png`, `sst2_accuracy.png`, `squad_exact_match.png`, `squad_f1.png`
- Training curve (loss over steps): `src/evals/results/plots/gpt2_lm_long_training.png`
- Training dynamics (loss, grad norm, LR): `src/evals/results/plots/gpt2_lm_long_dynamics.png`

## Regenerate plots
```
python src/evals/plot_eval_results.py \
  --inputs src/evals/results/base_gpt2_metrics.json src/evals/results/gpt2_eval.json src/evals/results/gpt2_lm_long_eval.json \
  --output-dir src/evals/results/plots
python src/evals/plot_training_curve.py \
  --trainer-state model/peft_hf_adapter_gpt2_lm_long/checkpoint-4000/trainer_state.json \
  --output src/evals/results/plots/gpt2_lm_long_training.png
python src/evals/plot_training_dynamics.py \
  --trainer-state model/peft_hf_adapter_gpt2_lm_long/checkpoint-4000/trainer_state.json \
  --output src/evals/results/plots/gpt2_lm_long_dynamics.png
```

## Hardware
- Training runs targeted RTX 3090 (24GB) as before.
- Latest quick evals above were executed on CPU (torch 2.9.1+cpu, Python 3.13.5) after adding SQuAD F1.
