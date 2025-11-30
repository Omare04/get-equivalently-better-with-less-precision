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
    return re.sub(r"={1,}", "", text)

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

def _preprocessWikitext(batch: Dict) -> Dict:
    cleaned_texts = []
    for text in batch['text']:
        text = clean_newlines(text)
        text = clean_brs(text)
        text = clean_equals(text) 
        text = clean_spaces(text)
        if text:
            cleaned_texts.append(text)
    
    return {'text': cleaned_texts} 

def _preprocessSST2(batch: Dict) -> Dict:
    cleaned_texts = [clean_spaces(t) for t in batch['sentence']]
    return {
        'text': cleaned_texts,
        'label': batch['label']
    }

def _preprocessSQuAD(batch: Dict) -> Dict:
    cleaned_contexts = [clean_newlines(clean_brs(c)) for c in batch['context']]
    cleaned_questions = [clean_spaces(q) for q in batch['question']]
    return {
        'context': cleaned_contexts,
        'question': cleaned_questions,
        'answers': batch['answers']
    }
