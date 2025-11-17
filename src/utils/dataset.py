## 
# 1 function that will get me a dataset with no preprocessing given a specific Name you know . 
# 
# 2 a preprocessing function for every specific dataset, since every dataset have some noise in it. i.e the wikidataset contain \n & ' ' and "====" which will cluter the training stuff
# also the sentiment analysis preprocessing is diffrent from a set to the other
# 
# we have to preprocess per batch
# # 
# 3 we need a function called         cleaned_dataset = preprocessDataset(batch_size, x, dataset_name) or getDataSetBatch(same parameters

# ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train") & SST-2 & IMDB training & SQuAD v1.1

from datasets import load_dataset, Dataset
from typing import List, Dict
from . import text_cleaners

# dataset_configs = {
#     "imdb": {
#         "path": "imdb",
#         "splits": ["train", "test", "unsupervised"],
#         "text_column": "text",
#         "label_column": "label"
#     },
#     "wikitext": {
#         "path": "wikitext",
#         "name": "wikitext-2-raw-v1",
#         "splits": ["train", "validation", "test"],
#         "text_column": "text",
#         "label_column": None
#     },
#     "sst2": {
#         "path": "glue",
#         "name": "sst2",
#         "splits": ["train", "validation", "test"],
#         "text_column": "sentence",
#         "label_column": "label"
#     },
#     "squad_v2": {
#         "path": "squad_v2",
#         "splits": ["train", "validation"],
#         "text_column": "context",
#         "label_column": None
#     }
# }
dataset_configs = {
    "imdb": {
        "path": "imdb",
        "splits": ["train", "test", "unsupervised"]
    },
    "wikitext": {
        "path": "wikitext",
        "name": "wikitext-2-raw-v1",
        "splits": ["train", "validation", "test"]
    },
    "sst2": {
        "path": "glue",
        "name": "sst2",
        "splits": ["train", "validation", "test"]
    },
    "squad_v2": {
        "path": "squad_v2",
        "splits": ["train", "validation"]
    }
}
def getRawDataset(dataset_name: str, split: str = "train") -> Dataset:
    if dataset_name not in dataset_configs:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Supported: {list(dataset_configs.keys())}"
        )
    
    config = dataset_configs[dataset_name]
    
    if split not in config["splits"]:
        raise ValueError(
            f"Invalid split '{split}' for {dataset_name}. "
            f"Available splits: {config['splits']}"
        )
    
    try:
        if "name" in config:
            dataset = load_dataset(config["path"], config["name"], split=split)
        else:
            dataset = load_dataset(config["path"], split=split)
        
        print(f"Loaded {dataset_name} ({split}): {len(dataset)}")
        return dataset
    
    except Exception as e:
        raise RuntimeError(f"Failed to load {dataset_name}: {str(e)}")
    
def preprocess(batch, dataset_name):
    if dataset_name == "imdb":
        return text_cleaners._preprocessIMDB(batch)
    elif dataset_name == "wikitext":
        return text_cleaners._preprocessWikitext(batch)
    elif dataset_name == "sst2":
        return text_cleaners._preprocessSST2(batch)
    elif dataset_name == "squad" or dataset_name == "squad_v2":
        return text_cleaners._preprocessSQuAD(batch)
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Supported: imdb, wikitext, sst2, squad"
        )

def getDatasetBatch(batch_size: int, dataset_name: str, split: str = "train") -> Dict:
    raw_dataset = getRawDataset(dataset_name, split)
    batch = raw_dataset.select(range(batch_size))

    if dataset_name == "wikitext":
        batch = [item for item in batch if item['text'].strip() != '']

    if dataset_name == "sst2":
        batch_dict = {k: [item[k] for item in batch] for k in batch[0]}
        cleaned_batch = preprocess(batch_dict, dataset_name)
    else:
        batch_dict = {k: [item[k] for item in batch] for k in batch[0]}
        cleaned_batch = preprocess(batch_dict, dataset_name)

    return cleaned_batch