"""
Plot training dynamics (loss, grad_norm, learning rate) from a Hugging Face Trainer state file.

Usage:
  python src/evals/plot_training_dynamics.py \
    --trainer-state model/peft_hf_adapter_gpt2_lm_long/checkpoint-4000/trainer_state.json \
    --output src/evals/results/plots/gpt2_lm_long_dynamics.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_logs(state_path: Path):
    state = json.load(state_path.open())
    return state.get("log_history", [])


def plot_dynamics(logs, output: Path):
    steps, losses, grad_norms, lrs, epochs = [], [], [], [], []
    for log in logs:
        if "loss" in log:
            steps.append(log.get("step"))
            losses.append(log.get("loss"))
            grad_norms.append(log.get("grad_norm"))
            lrs.append(log.get("learning_rate"))
            epochs.append(log.get("epoch"))

    if not steps:
        raise RuntimeError("No loss entries found in log_history.")

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    axes[0].plot(steps, losses, marker="o", markersize=2, linewidth=1, color="tab:blue")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, grad_norms, marker="o", markersize=2, linewidth=1, color="tab:orange")
    axes[1].set_ylabel("Grad Norm")
    axes[1].set_title("Gradient Norm")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, lrs, linewidth=1, color="tab:green")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_xlabel("Step")
    axes[2].set_title("Learning Rate Schedule")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.close()
    print(f"Saved training dynamics plot to {output}")


def main():
    parser = argparse.ArgumentParser(description="Plot loss/grad_norm/lr from trainer_state.json")
    parser.add_argument("--trainer-state", required=True, help="Path to trainer_state.json")
    parser.add_argument("--output", required=True, help="Output PNG path")
    args = parser.parse_args()

    state_path = Path(args.trainer_state)
    if not state_path.exists():
        raise FileNotFoundError(state_path)

    logs = load_logs(state_path)
    plot_dynamics(logs, Path(args.output))


if __name__ == "__main__":
    main()
