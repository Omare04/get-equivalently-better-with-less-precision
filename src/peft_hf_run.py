# PEFT LoRA training script using HF trainer for GPT-Neo-125M.
# This keeps the existing scripts untouched and offers a clean runner with sane defaults.
# Default dataset: wikitext-2-raw-v1.

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="EleutherAI/gpt-neo-125m")
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument(
        "--dataset-config",
        default="",
    )
    parser.add_argument("--output-dir", default="model/peft_hf_adapter")
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--train-limit", type=int, default=0,)
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=None,    )
    return parser.parse_args()


def load_lm_dataset(name: str, config: str, split: str, limit: int = 0):
    load_args = [name]
    if config:
        load_args.append(config)
    elif name == "wikitext":
        load_args.append("wikitext-2-raw-v1")

    ds = load_dataset(*load_args, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    ds = ds.filter(lambda x: x.get("text", "").strip() != "")
    return ds


def tokenize_fn(tokenizer, max_length: int):
    def _fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
        )

    return _fn


def infer_target_modules(model) -> list:
    """
    Auto-detect typical attention projection module names.
    """
    names = [n for n, _ in model.named_modules()]
    qkv = [n for n in names if any(k in n for k in ("q_proj", "k_proj", "v_proj"))]
    if qkv:
        return sorted({n.split(".")[-1] for n in qkv})
    gpt2_like = [n for n in names if "c_attn" in n]
    if gpt2_like:
        return ["c_attn", "c_proj"]
    raise ValueError("Could not infer target modules; please pass --target-modules explicitly.")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(args.model_name)
    base_model.to(device)

    target_modules = args.target_modules or infer_target_modules(base_model)

    lora_cfg = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    ds = load_lm_dataset(args.dataset, args.dataset_config, "train", limit=args.train_limit)
    tokenized = ds.map(
        tokenize_fn(tokenizer, args.max_length),
        batched=True,
        remove_columns=ds.column_names,
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    warmup_steps = int(args.max_steps * args.warmup_ratio)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        warmup_steps=warmup_steps,
        logging_steps=50,
        save_steps=args.max_steps,
        save_total_limit=1,
        report_to=[],
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
    )
    trainer.train()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved adapter and tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
