"""
Evaluate a PEFT LoRA adapter (e.g., from src/peft_lora.py) on lightweight tasks.
Metrics are saved to JSON under src/evals/results/.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Allow running as a script: add repo root and src/ to sys.path for absolute imports
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.evals.vanilla_lora_eval import run_all_evals


def parse_args():
    parser = argparse.ArgumentParser(description="Run evals on a PEFT LoRA adapter for GPT-Neo.")
    parser.add_argument(
        "--adapter-dir",
        required=True,
        help="Path to the adapter directory produced by src/peft_lora.py (contains adapter_model.safetensors).",
    )
    parser.add_argument(
        "--model-name",
        default="EleutherAI/gpt-neo-125m",
        help="Base model to load before attaching the adapter.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output JSON path. If empty, saved to src/evals/results/<adapter_name>_metrics.json",
    )
    parser.add_argument("--wikitext-samples", type=int, default=64)
    parser.add_argument("--imdb-samples", type=int, default=32)
    parser.add_argument("--sst2-samples", type=int, default=32)
    parser.add_argument("--squad-samples", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    adapter_path = Path(args.adapter_dir)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter dir not found: {adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name)

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.to(device)

    results = run_all_evals(
        model,
        tokenizer,
        sample_sizes={
            "wikitext": args.wikitext_samples,
            "imdb": args.imdb_samples,
            "sst2": args.sst2_samples,
            "squad": args.squad_samples,
        },
    )

    adapter_name = Path(args.adapter_dir).name
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path("src/evals/results") / f"{adapter_name}_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Eval results written to {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
