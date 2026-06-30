from typing import List, Dict
import vector_store
from sentence_transformers import CrossEncoder

TOP_K = 5
# Chroma cosine distance: 0 = identical, 2 = opposite. Empirically, chunks
# above this distance are not meaningfully related to the question.
MAX_DISTANCE = 1.0

# Load the CrossEncoder model once at module level to avoid reloading it on every query
_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def retrieve(question: str, top_k: int = TOP_K) -> List[Dict]:
    # Retrieve top 20 chunks from vector_store
    raw_hits = vector_store.query(question, n_results=20)
    
    # Distance threshold filter as a pre-filter before reranking
    relevant = [h for h in raw_hits if h["distance"] <= MAX_DISTANCE]
    
    if not relevant:
        return []
        
    # Score each (question, chunk_text) pair using CrossEncoder
    pairs = [(question, h["text"]) for h in relevant]
    scores = _reranker.predict(pairs)
    
    for h, score in zip(relevant, scores):
        h["rerank_score"] = float(score)
        
    # Sort by score descending
    relevant.sort(key=lambda x: x["rerank_score"], reverse=True)
    
    # Return only the top 5 (top_k)
    return relevant[:top_k]

def has_documents() -> bool:
    return vector_store.collection_stats()["total_chunks"] > 0

