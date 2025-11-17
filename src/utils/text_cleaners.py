from typing import List, Dict
import re

def clean_spaces(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def clean_brs(text: str) -> str:
    return re.sub(r"<br\s*/?>", " ", text)

def clean_newlines(text: str) -> str:
    return text.replace("\n", " ")

def clean_equals(text: str) -> str:
    return text.replace("=", "")



def _preprocessIMDB(batch: Dict) -> Dict:
    cleaned_texts = []
    for text in batch['text']:
        text = clean_brs(text)
        text = clean_newlines(text)
        text = clean_spaces(text)
        cleaned_texts.append(text)
    
    return {
        'text': cleaned_texts,
        'label': batch['label']
    }

def _preprocessWikitext(batch: Dict) -> List[str]:    
    return []

def _preprocessSST2(batch: Dict) -> List[str]:
    return []

def _preprocessSQuAD(batch: Dict) -> List[str]:
    return []