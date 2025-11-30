
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)


def build_model(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    return tokenizer, model


def apply_lora(model, r: int = 8, alpha: int = 16, dropout: float = 0.05):
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["q_proj", "k_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    lora_model = get_peft_model(model, lora_config)
    lora_model.print_trainable_parameters()
    return lora_model


def load_lm_dataset(name: str, split: str, config: str = "", field: str = "text", limit: int = 0):
    # Default to wikitext-2-raw-v1 if user provided bare "wikitext"
    load_args = [name]
    if config:
        load_args.append(config)

    if name == "wikitext" and not config:
        load_args.append("wikitext-2-raw-v1")

    ds = load_dataset(*load_args, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    ds = ds.filter(lambda x: x.get(field, "").strip() != "")
    return ds


def train_peft(
    model,
    tokenizer,
    dataset_name: str,
    dataset_config: str,
    output_dir: Path,
    max_steps: int = 200,
    batch_size: int = 4,
    lr: float = 2e-5,
    warmup_steps: int = 50,
    max_length: int = 128,
    train_limit: int = 0,
    weight_decay: float = 0.0,
    grad_accum: int = 1,
    seed: int = 42,
):
    ds = load_lm_dataset(dataset_name, "train", config=dataset_config, limit=train_limit)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
        )

    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)

    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        logging_steps=10,
        save_steps=max_steps,
        save_total_limit=1,
        report_to=[],
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=collator,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved PEFT LoRA adapter to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="PEFT LoRA example for GPT-Neo-125M.")
    parser.add_argument("--model-name", default="EleutherAI/gpt-neo-125m")
    parser.add_argument("--dataset", default="wikitext", help="HF dataset name (e.g., wikitext).")
    parser.add_argument("--dataset-config", default="", help="HF dataset config (e.g., wikitext-2-raw-v1).")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--train-limit", type=int, default=0, help="Limit training samples for quick runs.")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--output-dir", default="model/peft_lora_gptneo")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = build_model(args.model_name, device)
    model = apply_lora(model, r=args.r, alpha=args.alpha, dropout=args.dropout)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_peft(
        model,
        tokenizer,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        output_dir=output_dir,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        max_length=args.max_length,
        train_limit=args.train_limit,
        weight_decay=args.weight_decay,
        grad_accum=args.grad_accum,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
