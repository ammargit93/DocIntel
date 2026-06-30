from typing import List, Dict
import vector_store

TOP_K = 5
# Chroma cosine distance: 0 = identical, 2 = opposite. Empirically, chunks
# above this distance are not meaningfully related to the question.
MAX_DISTANCE = 1.0

def retrieve(question: str, top_k: int = TOP_K) -> List[Dict]:
    raw_hits = vector_store.query(question, n_results=top_k)
    relevant = [h for h in raw_hits if h["distance"] <= MAX_DISTANCE]
    return relevant

def has_documents() -> bool:
    return vector_store.collection_stats()["total_chunks"] > 0

