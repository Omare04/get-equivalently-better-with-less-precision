

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


### What is LoRA and Why Do We Use It?

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning (PEFT) method that allows us to adapt LLMs to new downstream tasks without retraining or modifying all of their parameters.

Instead of updating the model’s full set of weight matrices, which can contain billions of parameters, LoRA freezes the base model weights and introduces a pair of small, trainable low-rank adapter matrices. These matrices (denoted as (A) and (B)) approximate a weight update $$(\Delta W = B A)$$ of rank (r), where (r \ll d) (the original matrix dimension). The learned (W) is then scaled and added to the frozen base weights during training:

$$
W' = W + \alpha \frac{B A}{r}
$$

By learning only this small, low-rank update, LoRA significantly reduces the number of trainable parameters—making fine-tuning faster, cheaper, and more memory-efficient while maintaining strong performance on downstream tasks.



### Why Use PEFT Methods like LoRA and QLoRA?

Large models such as Qwen or LLaMA are already highly capable and generalize well across many tasks. However, adapting them to specific, smaller tasks (like domain-specific text generation or sentiment classification) traditionally requires massive GPU resources and time.

PEFT methods like LoRA and QLoRA solve this by:

* Freezing the base model parameters.
* Training only a small number of additional adapter weights.
* Reducing GPU memory requirements by orders of magnitude.
* Enabling efficient experimentation on consumer-grade GPUs.

In short: they let us fine-tune billion-parameter models using a single GPU making large model adaptation feasible for researchers and developers with limited hardware.

---

### Project Overview

In this project, we leverage Hugging Face’s PEFT library to fine-tune two instruction-tuned models from the Qwen family:

* `Qwen/Qwen2.5-3B-Instruct-AWQ`
* `Qwen/Qwen2.5-7B-Instruct-AWQ`

Each model will be fine-tuned using QLoRA and evaluated against its non-fine-tuned baseline using the metrics defined in the metrics table above.
This comparison will help us determine whether PEFT-based methods can achieve better task adaptation using a fraction of the parameters.

---

### Hardware Configuration

All experiments are conducted on an NVIDIA RTX 3090 GPU.
In FP32 precision, the RTX 3090 delivers approximately 35.58 TFLOPs of compute performance, as reported by multiple hardware benchmarks.

---

### Objective


The primary goal of these experiments is to evaluate whether PEFT (specifically LoRA and QLoRA) provides an effective and resource-efficient approach to fine-tuning large language models — achieving comparable or better performance using far fewer computational resources.