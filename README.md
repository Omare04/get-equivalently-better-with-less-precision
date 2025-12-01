# QLoRA Fine-Tuning of Qwen Models

This project explores parameter-efficient fine-tuning (PEFT) of Qwen and similarly sized causal LMs using Quantized Low-Rank Adaptation (QLoRA). The goal is to adapt models efficiently on a single consumer GPU while maintaining downstream quality on language modeling, sentiment, and QA tasks.

## Table of Contents
- [Overview](#overview)
- [Requirements & Setup](#requirements--setup)
- [Dataset Access](#dataset-access)
- [Train & Validate (LoRA/QLoRA)](#train--validate-loraqlora)
- [Run the Pre-Trained Adapter on Sample Test Slices](#run-the-pre-trained-adapter-on-sample-test-slices)
- [Evaluate & Plot](#evaluate--plot)
- [Datasets & Tasks](#datasets--tasks)
- [Results Snapshot](#results-snapshot)
- [Directory / PyTorch Source Layout](#directory--pytorch-source-layout)
- [PEFT Background](#peft-background)
- [Hardware Notes](#hardware-notes)
- [References](#references)

## Overview
- Base models: [`Qwen2.5-4B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-4B-Instruct) and [`Qwen2.5-8B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-8B-Instruct). Smaller experiments use GPT-2/GPT-Neo for quicker iterations.
- Fine-tuning method: QLoRA (4-bit NF4 quantization + low-rank adapters) via Hugging Face `peft`.
- Frameworks: PyTorch, `transformers`, `peft`, `datasets`, `bitsandbytes`.
- Objectives: improve perplexity on Wikitext, sentiment accuracy on IMDB/SST-2, and QA quality on SQuAD with minimal trainable parameters.

## Requirements & Setup
- Python 3.10+ and `pip`; Linux/macOS shells tested.
- GPU recommended: NVIDIA RTX 3090 (24GB) for QLoRA/LoRA training; CPU works for quick sample evals.
- Install libraries (includes `torch==2.9.1`, `transformers`, `peft`, `datasets`, CUDA 12 `bitsandbytes`):
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
- Hugging Face auth is only needed for gated models; default GPT-2/GPT-Neo runs are public.

## Dataset Access
All datasets stream from the Hugging Face Hub; the `datasets` library caches them locally. Direct links:
- Wikitext-2: https://huggingface.co/datasets/wikitext
- IMDB: https://huggingface.co/datasets/imdb
- SST-2: https://huggingface.co/datasets/glue/viewer/sst2
- SQuAD v1.1: https://huggingface.co/datasets/squad
- (Optional) AlpacaEval: https://huggingface.co/datasets/tatsu-lab/alpaca_eval

Pre-download the exact splits used in the sample evals:
```bash
python - <<'PY'
from datasets import load_dataset
load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
load_dataset("imdb", split="test")
load_dataset("glue", "sst2", split="validation")
load_dataset("squad", split="validation")
PY
```

## Train & Validate (LoRA/QLoRA)
1) Train a LoRA adapter (example: improved GPT-2 run, r=8, on Wikitext-2):
```bash
python src/peft_hf_run.py --model-name gpt2 \
  --dataset wikitext --dataset-config wikitext-2-raw-v1 \
  --max-steps 4000 --batch-size 8 --grad-accum 2 \
  --lr 1e-4 --warmup-ratio 0.1 --weight-decay 0.01 \
  --max-length 128 \
  --output-dir model/peft_hf_adapter_gpt2_lm_long
```
Outputs: adapter weights + tokenizer in `model/peft_hf_adapter_gpt2_lm_long/`. Swap `--model-name` to a Qwen checkpoint if you have sufficient VRAM.

2) Quick validation on the sample test slices (64 wikitext, 32 imdb, 32 sst2, 16 squad):
```bash
python src/evals/eval_peft_adapter.py \
  --adapter-dir model/peft_hf_adapter_gpt2_lm_long \
  --model-name gpt2 \
  --wikitext-samples 64 --imdb-samples 32 --sst2-samples 32 --squad-samples 16
```
Results are written to `src/evals/results/peft_hf_adapter_gpt2_lm_long_metrics.json`.

## Run the Pre-Trained Adapter on Sample Test Slices
Use the shipped adapter at `model/peft_hf_adapter_gpt2_lm_long`:
```bash
python src/evals/eval_peft_adapter.py \
  --adapter-dir model/peft_hf_adapter_gpt2_lm_long \
  --model-name gpt2
```
By default it evaluates the small sample slices above and saves `src/evals/results/peft_hf_adapter_gpt2_lm_long_metrics.json`. Inspect plots in `src/evals/results/plots/` for side-by-side comparisons.

## Evaluate & Plot
- Baseline (no adapter):
  ```bash
  python src/evals/eval_base_model.py --model-name gpt2 \
    --wikitext-samples 64 --imdb-samples 32 --sst2-samples 32 --squad-samples 16
  ```
- Regenerate plots:
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

## Datasets & Tasks

| Dataset        | Task                          | Metric                |
| -------------- | ----------------------------- | --------------------- |
| **SST-2**      | Sentiment classification      | Accuracy              |
| **SQuAD v1.1** | Extractive question answering | F1 Score, Exact Match |
| **AlpacaEval** | Instruction following         | Win Rate              |
| **Wikitext-2** | Language modeling (proxy)     | Perplexity, Loss      |

All datasets are loaded via the Hugging Face `datasets` library.

## Results Snapshot
- Settings: wikitext=64, imdb=32, sst2=32, squad=16; quick CPU pass (torch 2.9.1+cpu, Python 3.13.5).
- `base_gpt2_metrics.json`: ppl 54.26 (loss 3.99), IMDB 0.69, SST-2 0.44, SQuAD EM/F1 0.00 / 0.130.
- `gpt2_eval.json` (short LoRA, r=8): ppl 31.90, IMDB 0.59, SST-2 0.59, SQuAD EM/F1 0.00 / 0.102.
- `gpt2_lm_long_eval.json` (longer LoRA, r=8): ppl 31.64, IMDB 0.59, SST-2 0.56, SQuAD EM/F1 0.00 / 0.121. Plots live in `src/evals/results/plots/`.
- Run history: `base_EleutherAI_gpt-neo-125m_metrics.json` (sanity LM baseline), misses (`wt103_eval.json`, short `gpt2_eval.json`), and the final improved run (`gpt2_lm_long_eval.json`).

## Directory / PyTorch Source Layout
- `src/peft_hf_run.py`: HF Trainer-based QLoRA/LoRA training loop (PyTorch).  
- `src/peft_lora.py`: lightweight PyTorch/Trainer LoRA script.  
- `src/evals/*.py`: evaluation + plotting utilities (perplexity, sentiment, QA).  
- `src/utils/dataset.py`: dataset loaders and cleaners for Wikitext/IMDB/SST-2/SQuAD.  
- `model/`: saved adapters and checkpoints (including `peft_hf_adapter_gpt2_lm_long`).  
- `docs/`: reports and figures; `data/`: optional local caches.

## PEFT Background
- LoRA introduces small low-rank matrices (A, B) that approximate a weight update \(\Delta W = BA\) and are added to frozen base weights: \(W' = W + \alpha \frac{BA}{r}\). Training only these adapters cuts trainable parameters dramatically.
- QLoRA combines 4-bit NF4 quantization for the frozen backbone with LoRA adapters (often in bfloat16) to save VRAM while keeping quality competitive.
- Why PEFT: freeze most parameters, reduce memory/compute, and make large-model adaptation feasible on a single GPU.

## Hardware Notes
- Target GPU: NVIDIA RTX 3090 (24GB).  
- Typical precision: 4-bit NF4 backbone + bfloat16 adapters; optimizer: Paged AdamW; features: gradient checkpointing and rank ablations (r = 4, 8, 16).

## References
1. Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, NeurIPS 2023.  
2. Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022.  
3. Zhang et al., *Qwen2.5 Technical Report*, Alibaba Group, 2024.  
4. Houlsby et al., *Parameter-Efficient Transfer Learning for NLP*, ICML 2019.
