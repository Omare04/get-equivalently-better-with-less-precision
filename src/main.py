import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evals import vanilla_lora_eval  
from utils import dataset as dataset_utils  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_MODEL = "EleutherAI/gpt-neo-125m"
SUPPORTED_DATASETS = list(dataset_utils.dataset_configs.keys())


class LoRaHelper(nn.Module):
    def __init__(self, base_layer: nn.Linear, alpha: int, r: int = 4):
        super().__init__()

        self.base = base_layer
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        in_dim = self.base.in_features
        out_dim = self.base.out_features

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        ref = self.base.weight

        A = torch.empty((r, in_dim), device=ref.device, dtype=ref.dtype)
        A = torch.nn.init.normal_(A, mean=0.0, std=0.02)

        B = torch.zeros((out_dim, r), device=ref.device, dtype=ref.dtype)

        self.A = nn.Parameter(A)
        self.B = nn.Parameter(B)

    def forward(self, x):
        result = self.base(x)
        down_proj = x @ self.A.T
        up_proj = down_proj @ self.B.T

        return result + self.scaling * up_proj


def _collate_dicts(examples: Iterable[Dict]) -> Dict:
    keys = examples[0].keys()
    return {k: [ex[k] for ex in examples] for k in keys}


def _iter_attention_blocks(model) -> Iterable[Tuple[int, object]]:
    for idx, block in enumerate(model.transformer.h):
        yield idx, block.attn.attention


def add_lora_to_gptneo(model, alpha: int = 16, r: int = 4) -> None:
    for _, attn in _iter_attention_blocks(model):
        attn.k_proj = LoRaHelper(attn.k_proj, alpha=alpha, r=r)
        attn.q_proj = LoRaHelper(attn.q_proj, alpha=alpha, r=r)
        attn.v_proj = LoRaHelper(attn.v_proj, alpha=alpha, r=r)
        attn.out_proj = LoRaHelper(attn.out_proj, alpha=alpha, r=r)


def save_model_target_params(
    target_model,
    stage: str = "base_model_attention_proj_weights",
    output_root: Path = Path("model/lora"),
) -> Path:
    """Persist base and LoRA weights so runs can be swapped in and out."""
    stage_path = output_root / stage
    stage_path.mkdir(parents=True, exist_ok=True)

    meta: Dict[str, Dict[str, int]] = {}

    for idx, attn in _iter_attention_blocks(target_model):
        block_dir = stage_path / f"block_{idx}"
        block_dir.mkdir(parents=True, exist_ok=True)

        for proj_name in ["k_proj", "q_proj", "v_proj", "out_proj"]:
            layer = getattr(attn, proj_name)

            if hasattr(layer, "base"):
                torch.save(layer.base.weight.detach().cpu(), block_dir / f"{proj_name}.base.pt")
                torch.save(layer.A.detach().cpu(), block_dir / f"{proj_name}.A.pt")
                torch.save(layer.B.detach().cpu(), block_dir / f"{proj_name}.B.pt")
                meta[proj_name] = {"alpha": layer.alpha, "r": layer.r}
            else:
                torch.save(layer.weight.detach().cpu(), block_dir / f"{proj_name}.base.pt")

    with (stage_path / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return stage_path


def load_lora_weights(
    target_model,
    stage: str,
    load_base: bool = False,
    output_root: Path = Path("model/lora"),
) -> Path:
    """Load LoRA weights (and optionally base projections) from disk."""
    stage_path = Path(stage)
    if not stage_path.exists():
        stage_path = output_root / stage

    if not stage_path.exists():
        raise FileNotFoundError(f"Could not find saved weights at {stage_path}")

    for idx, attn in _iter_attention_blocks(target_model):
        block_dir = stage_path / f"block_{idx}"
        for proj_name in ["k_proj", "q_proj", "v_proj", "out_proj"]:
            layer = getattr(attn, proj_name)
            base_path = block_dir / f"{proj_name}.base.pt"
            a_path = block_dir / f"{proj_name}.A.pt"
            b_path = block_dir / f"{proj_name}.B.pt"

            if load_base and base_path.exists():
                target = layer.base if hasattr(layer, "base") else layer
                target.weight.data.copy_(torch.load(base_path, map_location=device))

            if hasattr(layer, "A") and a_path.exists() and b_path.exists():
                layer.A.data.copy_(torch.load(a_path, map_location=device))
                layer.B.data.copy_(torch.load(b_path, map_location=device))

    return stage_path


def list_saved_stages(output_root: Path = Path("model/lora")) -> Iterable[str]:
    if not output_root.exists():
        return []
    return sorted([p.name for p in output_root.iterdir() if p.is_dir()])


def _tokenize_cleaned_batch(
    cleaned: Dict,
    dataset_name: str,
    tokenizer,
    max_length: int,
) -> Optional[Tuple[Dict[str, torch.Tensor], torch.Tensor]]:
    if dataset_name.startswith("squad"):
        tokenized = tokenizer(
            cleaned["question"],
            text_pair=cleaned["context"],
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
        ).to(device)
    else:
        if not cleaned.get("text"):
            return None
        tokenized = tokenizer(
            cleaned["text"],
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
        ).to(device)

    labels = tokenized["input_ids"].clone()
    labels[tokenized["attention_mask"] == 0] = -100
    return tokenized, labels


def _build_dataloader(
    dataset_name: str,
    split: str,
    batch_size: int,
    limit: Optional[int] = None,
) -> DataLoader:
    ds = dataset_utils.getRawDataset(dataset_name, split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=_collate_dicts)


def train_lora_model(
    model,
    tokenizer,
    dataset_name: str,
    epochs: int = 1,
    batch_size: int = 2,
    lr: float = 2e-5,
    max_length: int = 128,
    limit: Optional[int] = None,
    log_interval: int = 25,
    grad_clip: float = 1.0,
    warmup_steps: int = 0,
) -> float:
    dataloader = _build_dataloader(dataset_name, "train", batch_size, limit)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    last_loss = 0.0
    total_steps = epochs * len(dataloader)
    scheduler = None
    if warmup_steps and total_steps:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"epoch {epoch + 1}/{epochs}")

        for step, raw_batch in enumerate(pbar):
            cleaned = dataset_utils.preprocess(raw_batch, dataset_name)
            tokenized_labels = _tokenize_cleaned_batch(cleaned, dataset_name, tokenizer, max_length)
            if tokenized_labels is None:
                continue

            tokenized, labels = tokenized_labels
            outputs = model(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
                labels=labels,
            )

            loss = outputs.loss
            last_loss = float(loss.detach().cpu())

            optimizer.zero_grad()
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
            optimizer.step()
            if scheduler:
                scheduler.step()
            pbar.set_postfix(loss=last_loss)

            if log_interval and (step + 1) % log_interval == 0:
                print(f"[epoch {epoch + 1} step {step + 1}] loss={last_loss:.4f}", flush=True)

    return last_loss


def build_model_and_tokenizer(model_name: str = DEFAULT_MODEL):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    return tokenizer, model


def run_job(args):
    print(f"Using device: {device}")
    tokenizer, model = build_model_and_tokenizer(args.model_name)

    if args.save_base and "base_model_attention_proj_weights" not in list_saved_stages():
        save_model_target_params(model, stage="base_model_attention_proj_weights")

    add_lora_to_gptneo(model, alpha=args.alpha, r=args.rank)

    if args.load_stage:
        load_lora_weights(model, args.load_stage, load_base=args.load_base)
        print(f"Loaded LoRA weights from {args.load_stage}")

    final_loss = None
    if not args.eval_only:
        final_loss = train_lora_model(
            model,
            tokenizer,
            dataset_name=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_length=args.max_length,
            limit=args.train_limit,
            log_interval=args.log_interval,
            grad_clip=args.grad_clip,
            warmup_steps=args.warmup_steps,
        )
        stage_path = save_model_target_params(model, stage=args.run_name)
        print(f"Saved LoRA weights to {stage_path}")
        print(f"Final training loss: {final_loss}")

    eval_results = None
    if args.run_evals:
        eval_results = vanilla_lora_eval.run_all_evals(
            model,
            tokenizer,
            sample_sizes={
                "wikitext": args.eval_wikitext_samples,
                "imdb": args.eval_imdb_samples,
                "sst2": args.eval_sst2_samples,
                "squad": args.eval_squad_samples,
            },
        )
        results_path = Path("src/evals/results") / f"{args.run_name}_metrics.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with results_path.open("w", encoding="utf-8") as f:
            json.dump(eval_results, f, indent=2)
        print(f"Evals written to {results_path}")

    return final_loss, eval_results


def parse_args():
    parser = argparse.ArgumentParser(description="Vanilla LoRA trainer for GPT-Neo.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default="wikitext", choices=SUPPORTED_DATASETS)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--rank", type=int, default=4, help="LoRA rank r.")
    parser.add_argument("--alpha", type=int, default=16, help="LoRA alpha scaling.")
    parser.add_argument("--run-name", default="exp_latest")
    parser.add_argument("--train-limit", type=int, default=0, help="Limit training samples for quick runs.")
    parser.add_argument("--load-stage", default="", help="Stage name or path to load LoRA weights from.")
    parser.add_argument("--load-base", action="store_true", help="Also load base projection weights from stage.")
    parser.add_argument("--save-base", action="store_true", help="Save base projection weights before training.")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and only run evals.")
    parser.add_argument("--run-evals", action="store_true", help="Run downstream evals after training.")
    parser.add_argument("--log-interval", type=int, default=25, help="Steps between loss prints during training.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm (0 to disable).")
    parser.add_argument("--warmup-steps", type=int, default=0, help="Linear LR warmup steps (0 disables scheduler).")
    parser.add_argument("--eval-wikitext-samples", type=int, default=64)
    parser.add_argument("--eval-imdb-samples", type=int, default=32)
    parser.add_argument("--eval-sst2-samples", type=int, default=32)
    parser.add_argument("--eval-squad-samples", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_job(args)
