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


if __name__ == "__main__":
    sst2, squad, alpaca_eval = load_benchmark_datasets()
    print("Loaded datasets:")
    print("SST-2:", sst2)
    print("SQuAD v1.1:", squad)
    print("AlpacaEval:", alpaca_eval)