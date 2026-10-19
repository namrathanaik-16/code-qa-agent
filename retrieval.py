import json 
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

CHUNKS_PATH = Path("data/chunks.jsonl")

STOPWORDS={
    "a","an","the","is","are","was","were","be","to","of","in","on","for",
    "and","or","it","its","this","that","with","as","by","at","some","how",
    "what","where","which","who","why","does","do","did","can","i","you",
    "my","me","use","used","using","fastapi",
}
def tokenize(text:str) -> list:
    """Split text into search tokens, understanding code naming styles.
    
    'get_user_by_id' -> get_user_by_id, get, user, by ,id
    'OAuth2PasswordBearer' -> oauth2passwordbearer, o, auth, 2, password, bearer 
    """
    tokens=[]
    for word in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text):
        whole = word.lower().strip("_")
        parts = [p.lower() for p in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", word.replace("_", " "))]
        if len(parts) > 1 and whole:
            tokens.append(whole)
        tokens.extend(parts)
    return[t for t in tokens if len(t) > 1 and t not in STOPWORDS]

def load_chunks(path: Path = CHUNKS_PATH) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]
    
class KeywordRetriever:
    """BM25 keyword search pver code and doc chunks."""
    
    def __init__(self, chunks: list):
        self.chunks = chunks
        #file path and name are indexed too; the name is repeated to weight it higher.
        docs = [tokenize(f"{c['file']} {c['name']} {c['name']} {c['text']}") for c in chunks]
        self.bm25 = BM25Okapi(docs)
        
    def search(self, query: str, k: int=5)->list:
        """Return the top k(score, chunk) pairs, best first."""
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key = lambda i: scores[i], reverse=True)[:k]
        return [(float(scores[i]),self.chunks[i]) for i in ranked if scores[i] > 0]