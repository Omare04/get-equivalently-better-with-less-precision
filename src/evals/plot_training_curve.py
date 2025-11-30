"""
Plot training loss over steps/epochs from a Hugging Face Trainer state file.

Usage:
  python src/evals/plot_training_curve.py \
    --trainer-state model/peft_hf_adapter_gpt2_lm_long/checkpoint-4000/trainer_state.json \
    --output src/evals/results/plots/gpt2_lm_long_training.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot training loss from trainer_state.json")
    parser.add_argument("--trainer-state", required=True, help="Path to trainer_state.json")
    parser.add_argument("--output", required=True, help="Output PNG path")
    args = parser.parse_args()

    state_path = Path(args.trainer_state)
    if not state_path.exists():
        raise FileNotFoundError(state_path)

    state = json.load(state_path.open())
    logs = state.get("log_history", [])

    steps, losses, epochs = [], [], []
    for log in logs:
        if "loss" in log:
            steps.append(log.get("step"))
            losses.append(log.get("loss"))
            epochs.append(log.get("epoch"))

    if not steps:
        raise RuntimeError("No loss entries found in log_history.")

    plt.figure(figsize=(8, 4))
    plt.plot(steps, losses, marker="o", markersize=2, linewidth=1)
    plt.xlabel("Step")
    plt.ylabel("Training loss")
    plt.title("Training loss over steps")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved training curve to {out_path}")


if __name__ == "__main__":
    main()
