import json
import math
import os
from pathlib import Path
import re
import string
from collections import Counter
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from utils import dataset as dataset_utils


def _collate_dicts(examples: List[Dict]) -> Dict:
    keys = examples[0].keys()
    return {k: [ex[k] for ex in examples] for k in keys}


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _normalize_answer(text: str) -> str:
    """
    Standard SQuAD-style normalization: lowercase, strip punctuation/articles, and squeeze whitespace.
    """
    text = text.lower()
    text = "".join(ch if ch not in string.punctuation else " " for ch in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text


def _f1_score(prediction: str, gold_texts: List[str]) -> float:
    """
    Compute token-level F1 between a prediction and a list of gold answers.
    Uses the max F1 across gold answers, as in SQuAD evaluation.
    """
    pred_tokens = _normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0

    best_f1 = 0.0
    for gold in gold_texts:
        gold_tokens = _normalize_answer(gold).split()
        if not gold_tokens:
            continue

        common = Counter(pred_tokens) & Counter(gold_tokens)
        overlap = sum(common.values())
        if overlap == 0:
            f1 = 0.0
        else:
            precision = overlap / len(pred_tokens)
            recall = overlap / len(gold_tokens)
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

        best_f1 = max(best_f1, f1)

    return best_f1


def perplexity_eval(
    model,
    tokenizer,
    dataset_name: str = "wikitext",
    split: str = "validation",
    num_samples: int = 64,
    max_length: int = 128,
) -> Dict[str, float]:
    """Compute a lightweight perplexity estimate on a held-out split."""
    ds = dataset_utils.getRawDataset(dataset_name, split)
    if num_samples:
        ds = ds.select(range(min(num_samples, len(ds))))

    dataloader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=_collate_dicts)
    losses = []
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            cleaned = dataset_utils.preprocess(batch, dataset_name)
            if not cleaned.get("text"):
                continue

            tokenized = tokenizer(
                cleaned["text"],
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=max_length,
            ).to(model.device)

            labels = tokenized["input_ids"].clone()
            labels[tokenized["attention_mask"] == 0] = -100

            outputs = model(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
                labels=labels,
            )
            losses.append(outputs.loss.detach().cpu())

    if not losses:
        return {"perplexity": float("nan"), "loss": float("nan")}

    loss = torch.stack(losses).mean()
    return {"perplexity": math.exp(loss.item()), "loss": loss.item()}


def _decode_sentiment_label(text: str) -> Optional[int]:
    text = text.lower()
    if "positive" in text:
        return 1
    if "negative" in text:
        return 0
    return None


def sentiment_eval(
    model,
    tokenizer,
    dataset_name: str = "imdb",
    split: str = "test",
    num_samples: int = 64,
    max_length: int = 256,
) -> Dict[str, float]:
    """Estimate sentiment accuracy via simple prompt-and-generate."""
    ds = dataset_utils.getRawDataset(dataset_name, split)
    if num_samples:
        ds = ds.select(range(min(num_samples, len(ds))))

    dataloader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=_collate_dicts)
    tokenizer.pad_token = tokenizer.eos_token

    correct = 0
    total = 0
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            cleaned = dataset_utils.preprocess(batch, dataset_name)

            for text, label in zip(cleaned["text"], cleaned["label"]):
                prompt = f"Review: {text}\nSentiment (positive/negative):"
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=max_length,
                ).to(model.device)

                generated = model.generate(
                    **inputs,
                    max_new_tokens=3,
                    pad_token_id=tokenizer.eos_token_id,
                )
                new_tokens = generated[0][inputs["input_ids"].shape[1]:]
                decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
                pred = _decode_sentiment_label(decoded)

                if pred is None:
                    logits = model(**inputs).logits[0, -1]
                    pos_id = tokenizer.encode(" positive", add_special_tokens=False)[0]
                    neg_id = tokenizer.encode(" negative", add_special_tokens=False)[0]
                    pred = 1 if logits[pos_id] >= logits[neg_id] else 0

                correct += int(pred == label)
                total += 1

    return {"accuracy": correct / total if total else 0.0}


def squad_eval(
    model,
    tokenizer,
    dataset_name: str = "squad",
    split: str = "validation",
    num_samples: int = 32,
    max_length: int = 384,
    max_answer_tokens: int = 32,
) -> Dict[str, float]:
    """Rudimentary exact-match evaluation for extractive QA."""
    ds = dataset_utils.getRawDataset(dataset_name, split)
    if num_samples:
        ds = ds.select(range(min(num_samples, len(ds))))

    dataloader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=_collate_dicts)
    tokenizer.pad_token = tokenizer.eos_token

    exact = 0
    f1_total = 0.0
    total = 0
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            cleaned = dataset_utils.preprocess(batch, dataset_name)

            for context, question, answers in zip(
                cleaned["context"], cleaned["question"], cleaned["answers"]
            ):
                prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=max_length,
                ).to(model.device)

                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_answer_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                )
                new_tokens = generated[0][inputs["input_ids"].shape[1]:]
                prediction = tokenizer.decode(new_tokens, skip_special_tokens=True)

                pred_norm = _normalize_answer(prediction)
                gold_norms = [_normalize_answer(ans) for ans in answers["text"]]

                if pred_norm in gold_norms:
                    exact += 1
                f1_total += _f1_score(prediction, answers["text"])
                total += 1

    return {
        "exact_match": exact / total if total else 0.0,
        "f1": f1_total / total if total else 0.0,
    }


def alpaca_eval_preview(
    model,
    tokenizer,
    split: str = "eval",
    num_samples: int = 16,
    max_length: int = 512,
    output_path: Optional[Path] = None,
) -> Dict:
    """
    Generate responses on AlpacaEval prompts. This does not compute a win rate
    (requires external judge) but writes generations for downstream scoring.
    """
    try:
        ds = load_dataset("tatsu-lab/alpaca_eval", split=split)
    except Exception as exc:  # pragma: no cover - dataset availability may vary
        return {"error": str(exc)}

    if num_samples:
        ds = ds.select(range(min(num_samples, len(ds))))

    dataloader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=_collate_dicts)
    tokenizer.pad_token = tokenizer.eos_token

    generations = []
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            instruction = batch.get("instruction", [""])[0]
            input_text = batch.get("input", [""])[0]

            prompt = f"Instruction: {instruction}"
            if input_text:
                prompt += f"\nInput: {input_text}"
            prompt += "\nResponse:"

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length,
            ).to(model.device)

            generated = model.generate(
                **inputs,
                max_new_tokens=128,
                pad_token_id=tokenizer.eos_token_id,
            )
            new_tokens = generated[0][inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)

            generations.append(
                {
                    "prompt": prompt,
                    "response": response.strip(),
                    "reference": batch.get("output", [""])[0],
                }
            )

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for row in generations:
                f.write(json.dumps(row) + "\n")

    return {"samples": len(generations), "preview_path": str(output_path) if output_path else ""}


def run_all_evals(
    model,
    tokenizer,
    sample_sizes: Optional[Dict[str, int]] = None,
) -> Dict[str, Dict[str, float]]:
    sample_sizes = sample_sizes or {}

    results: Dict[str, Dict[str, float]] = {
        "wikitext": perplexity_eval(
            model,
            tokenizer,
            num_samples=sample_sizes.get("wikitext", 64),
        ),
        "imdb": sentiment_eval(
            model,
            tokenizer,
            dataset_name="imdb",
            num_samples=sample_sizes.get("imdb", 32),
        ),
        "sst2": sentiment_eval(
            model,
            tokenizer,
            dataset_name="sst2",
            split="validation",
            num_samples=sample_sizes.get("sst2", 32),
        ),
        "squad": squad_eval(
            model,
            tokenizer,
            dataset_name="squad",
            split="validation",
            num_samples=sample_sizes.get("squad", 16),
        ),
    }

    results["meta"] = _get_hardware_info()

    return results


def _get_hardware_info() -> Dict[str, object]:
    info: Dict[str, object] = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        try:
            device_idx = torch.cuda.current_device()
            prop = torch.cuda.get_device_properties(device_idx)
            info.update(
                {
                    "gpu_name": prop.name,
                    "total_memory_gb": round(prop.total_memory / (1024 ** 3), 2),
                    "compute_capability": f"{prop.major}.{prop.minor}",
                    "driver_version": torch.version.cuda,
                }
            )
        except Exception:
            info["gpu_name"] = "unknown"
    info["torch_version"] = torch.__version__
    info["python_version"] = os.sys.version
    return info
