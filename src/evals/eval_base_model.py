"""
Evaluate a base causal LM (no adapter) on the lightweight metrics we defined:
 - Wikitext perplexity
 - IMDB/SST2 sentiment accuracy via prompting
 - SQuAD exact match via prompted QA
Metrics are saved to JSON under src/evals/results/.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Make sure repo root and src/ are importable when run as a script
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.evals.vanilla_lora_eval import run_all_evals  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a base GPT-Neo (or other) model on lightweight metrics.")
    parser.add_argument("--model-name", default="EleutherAI/gpt-neo-125m", help="HF model id to evaluate.")
    parser.add_argument(
        "--output",
        default="",
        help="Output JSON path. If empty, saved to src/evals/results/base_<model_name_sanitized>_metrics.json",
    )
    parser.add_argument("--wikitext-samples", type=int, default=64)
    parser.add_argument("--imdb-samples", type=int, default=32)
    parser.add_argument("--sst2-samples", type=int, default=32)
    parser.add_argument("--squad-samples", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
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

    # Derive default output path
    if args.output:
        out_path = Path(args.output)
    else:
        safe_name = args.model_name.replace("/", "_")
        out_path = Path("src/evals/results") / f"base_{safe_name}_metrics.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Eval results written to {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
