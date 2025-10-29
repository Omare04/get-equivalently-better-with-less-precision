import random
from datasets import load_dataset

def load_benchmark_datasets():
    """
    Load SST-2, SQuAD v1.1, and AlpacaEval datasets into memory
    using Hugging Face's `datasets` library.

    Returns:
        tuple: (sst2_dataset, squad_dataset, alpaca_eval_dataset)
    """

    sst2 = load_dataset("glue", "sst2", keep_in_memory=True)

    squad = load_dataset("squad", keep_in_memory=True)

    alpaca_eval = load_dataset("tatsu-lab/alpaca_eval", keep_in_memory=True)

    return sst2, squad, alpaca_eval

def get_dataset_stats(dataset, name="dataset"):
    print(f"\nDataset: {name}")
    for split in dataset.keys():
        print(f"  • Split: {split}")
        print(f"    - Num examples: {len(dataset[split])}")
        print(f"    - Features: {list(dataset[split].features.keys())[:8]}...")


def sample_dataset(dataset, n=3, split="train"):
    print(f"\nRandom samples from {split} split:")
    for i in random.sample(range(len(dataset[split])), min(n, len(dataset[split]))):
        print(dataset[split][i])

def tokenize_dataset(dataset, tokenizer, text_field="text", max_length=128):
    def tokenize_fn(examples):
        return tokenizer(examples[text_field], truncation=True, padding="max_length", max_length=max_length)
    tokenized = dataset.map(tokenize_fn, batched=True)
    print(f"Tokenized dataset ({len(tokenized)} examples).")
    return tokenized

if __name__ == "__main__":
    sst2, squad, alpaca_eval = load_benchmark_datasets()
    print("Loaded datasets:")
    print("SST-2:", sst2)
    print("SQuAD v1.1:", squad)
    print("AlpacaEval:", alpaca_eval)