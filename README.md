

# QLoRA Fine-Tuning of Qwen Models

This project explores Parameter-Efficient Fine-Tuning (PEFT) of large language models using Quantized Low-Rank Adaptation (QLoRA). The goal is to reproduce and extend the QLoRA technique on Qwen2.5 models to enable efficient adaptation on consumer GPUs without sacrificing model performance.

---

## Overview

* **Base Models:**

  * [`Qwen2.5-4B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-4B-Instruct)
  * [`Qwen2.5-8B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-8B-Instruct)

* **Fine-Tuning Method:**

  * **QLoRA (Quantized Low-Rank Adaptation)** combining 4-bit quantization (`NF4`) with low-rank adapters (`LoRA`) for efficient fine-tuning.

* **Frameworks & Libraries:**

  * [Hugging Face PEFT](https://github.com/huggingface/peft)
  * [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
  * [Transformers](https://huggingface.co/docs/transformers)
  * [PyTorch](https://pytorch.org/)

---

## Datasets

The project evaluates QLoRA fine-tuning across three key NLP tasks:

| Dataset        | Task                          | Metric                |
| -------------- | ----------------------------- | --------------------- |
| **SST-2**      | Sentiment classification      | Accuracy              |
| **SQuAD v1.1** | Extractive question answering | F1 Score, Exact Match |
| **AlpacaEval** | Instruction following         | Win Rate              |

All datasets are loaded via the Hugging Face `datasets` library.

---

## Evaluation Metrics

**Quantitative:**

* Accuracy (SST-2)
* F1 Score & Exact Match (SQuAD)
* Win Rate (AlpacaEval)
* GPU memory usage (GB)
* Number and percentage of trainable parameters
* Training throughput (tokens/sec)

**Qualitative:**

* Rank vs. accuracy trade-off plots
* Performance vs. VRAM scaling visualizations
* Validation loss curves comparing QLoRA vs. full fine-tuning
* Example-based generation comparisons

---

## Directory Structure

```
project-root/
│
├── evals/
│   ├── evaluation_scripts/
│   └── results/
│   • Contains evaluation scripts, benchmark notebooks, and metric computation utilities.
│
├── src/
│   ├── train_qlora.py
│   ├── model_utils.py
│   └── config/
│   • Core source code for model loading, QLoRA configuration, training loops, and adapter setup.
│
├── docs/
│   ├── proposal.pdf
│   └── figures/
│   • Documentation files, reports, visualizations, and project write-ups.
│
├── data/
│   ├── sst2/
│   ├── squad/
│   └── alpacaeval/
│   • Processed datasets and cached tokenized data for training and evaluation.
│
└── README.md
```

---

## ⚙️ Hardware & Configuration

* **GPU:** NVIDIA RTX 3090 (24GB VRAM)
* **RAM:** 32GB DDR5
* **CPU** Intel Ulta 7 265KF

* **Precision:**
  * 4-bit NF4 quantization for base model
  * bfloat16 precision for LoRA adapters
* **Optimizer:** Paged AdamW
* **Features:** Gradient checkpointing, rank ablations (r = 4, 8, 16)

---

## References

1. Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, NeurIPS 2023.
2. Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022.
3. Zhang et al., *Qwen2.5 Technical Report*, Alibaba Group, 2024.
4. Houlsby et al., *Parameter-Efficient Transfer Learning for NLP*, ICML 2019.
