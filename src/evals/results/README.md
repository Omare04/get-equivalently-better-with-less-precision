# Eval Results, How to Run, and Reproduction Guide

This repo experiments with parameter-efficient fine-tuning (LoRA/QLoRA) of compact causal LMs (GPT-2 and GPT-Neo) to improve language modeling, sentiment, and QA quality without full-model updates. Everything runs in PyTorch with Hugging Face `transformers`, `peft`, and `datasets`.

## Requirements
- Python 3.10+ and `pip`; Linux/macOS shells tested.  
- GPU recommended: RTX 3090 24GB (used for training). CPU works for the quick sample evals.  
- Install libraries from `requirements.txt` (includes `torch==2.9.1`, `transformers`, `peft`, `datasets`, `bitsandbytes` for CUDA 12 builds):
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
- Hugging Face auth is only needed if you pull gated models; the default GPT-2/GPT-Neo models are public.

## Dataset access (download links)
All datasets stream from the Hugging Face Hub; `datasets` caches them automatically. Direct links:
- Wikitext-2: https://huggingface.co/datasets/wikitext  
- IMDB reviews: https://huggingface.co/datasets/imdb  
- SST-2: https://huggingface.co/datasets/glue/viewer/sst2  
- SQuAD v1.1: https://huggingface.co/datasets/squad  
- (Optional) AlpacaEval prompts: https://huggingface.co/datasets/tatsu-lab/alpaca_eval

To pre-download the exact splits used in the evals:
```bash
python - <<'PY'
from datasets import load_dataset
load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
load_dataset("imdb", split="test")
load_dataset("glue", "sst2", split="validation")
load_dataset("squad", split="validation")
PY
```

## Train and validate a LoRA adapter (PyTorch code in `src/`)
1) Train (matching the improved run, r=8, on wikitext-2):
```bash
python src/peft_hf_run.py --model-name gpt2 \
  --dataset wikitext --dataset-config wikitext-2-raw-v1 \
  --max-steps 4000 --batch-size 8 --grad-accum 2 \
  --lr 1e-4 --warmup-ratio 0.1 --weight-decay 0.01 \
  --max-length 128 \
  --output-dir model/peft_hf_adapter_gpt2_lm_long
```
Outputs: adapter weights + tokenizer under `model/peft_hf_adapter_gpt2_lm_long/`.

2) Quick validation on the sample test slices (64 wikitext, 32 imdb, 32 sst2, 16 squad):
```bash
python src/evals/eval_peft_adapter.py \
  --adapter-dir model/peft_hf_adapter_gpt2_lm_long \
  --model-name gpt2 \
  --wikitext-samples 64 --imdb-samples 32 --sst2-samples 32 --squad-samples 16
```
Results land in `src/evals/results/peft_hf_adapter_gpt2_lm_long_metrics.json`.

3) Baseline check (no adapter):
```bash
python src/evals/eval_base_model.py --model-name gpt2 \
  --wikitext-samples 64 --imdb-samples 32 --sst2-samples 32 --squad-samples 16
```

4) Plotting:
```bash
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

## Run the provided pre-trained model on the sample test dataset
Use the shipped adapter at `model/peft_hf_adapter_gpt2_lm_long`:
```bash
python src/evals/eval_peft_adapter.py \
  --adapter-dir model/peft_hf_adapter_gpt2_lm_long \
  --model-name gpt2
```
By default it evaluates the small sample slices above (wikitext-2 val, IMDB test, SST-2 val, SQuAD val) and writes `src/evals/results/peft_hf_adapter_gpt2_lm_long_metrics.json`. Inspect plots in `src/evals/results/plots/` for side-by-side comparisons.

## PyTorch source layout
- `src/peft_hf_run.py`: HF Trainer-based LoRA/QLoRA training loop.  
- `src/peft_lora.py`: lightweight PyTorch/Trainer LoRA training without extra helpers.  
- `src/evals/*.py`: evaluation and plotting utilities (perplexity, sentiment, QA).  
- `src/utils/dataset.py`: dataset loaders/cleaners for Wikitext/IMDB/SST-2/SQuAD.  
- `model/`: saved adapters and checkpoints. All training/eval code is PyTorch-first.

## Current eval snapshots (quick view)
- Settings: wikitext=64, imdb=32, sst2=32, squad=16; quick CPU pass (torch 2.9.1+cpu, Python 3.13.5).
- `base_gpt2_metrics.json`: ppl 54.26 (loss 3.99), IMDB 0.69, SST-2 0.44, SQuAD EM/F1 0.00 / 0.130.
- `gpt2_eval.json` (short LoRA, r=8): ppl 31.90, IMDB 0.59, SST-2 0.59, SQuAD EM/F1 0.00 / 0.102.
- `gpt2_lm_long_eval.json` (longer LoRA, r=8): ppl 31.64, IMDB 0.59, SST-2 0.56, SQuAD EM/F1 0.00 / 0.121. Plots live in `src/evals/results/plots/`.

## Run history (for context)
- Baselines: `base_EleutherAI_gpt-neo-125m_metrics.json` (sanity LM baseline), `base_gpt2_metrics.json` (main comparison point).  
- Misses: `wt103_eval.json` (GPT-Neo LoRA on WikiText-103; unstable, worse perplexity), `gpt2_eval.json` (short GPT-2 LoRA; not enough steps, LR too high).  
- Final improved run: `gpt2_lm_long_eval.json` from the training command above. Key choices: wikitext-2 corpus, lower LR + weight decay + warmup, grad accumulation for effective batch size on a single 3090.

## Hardware notes
- Training targeted an RTX 3090 (24GB).  
- Quick evals were run on CPU; GPU automatically used if available (`torch.cuda.is_available()`).
