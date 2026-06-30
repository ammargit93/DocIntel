from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions

PERSIST_DIR = "/chroma_db"
COLLECTION_NAME = "documents"

# Local, fast embedding model -> chosen for latency (see module docstring).
_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

_client = chromadb.PersistentClient(path=PERSIST_DIR)

def get_collection():
    return _client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

def add_chunks(chunks: List[Any]) -> int:
    """Embed and store a list of ingest.Chunk objects. Returns count stored."""
    if not chunks:
        return 0
    collection = get_collection()
    collection.add(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[
            {"source": c.source, "page": c.page, "chunk_index": c.chunk_index}
            for c in chunks
        ],
    )
    return len(chunks)

def query(question: str, n_results: int = 5) -> List[Dict]:
    """Return top-k relevant chunks with their metadata + distance score."""
    collection = get_collection()
    if collection.count() == 0:
        return []
    n_results = min(n_results, collection.count())
    results = collection.query(query_texts=[question], n_results=n_results)
    hits = []
    if not results or not results.get("ids") or not results["ids"][0]:
        return []
    for i in range(len(results["ids"][0])):
        hits.append(
            {
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if results.get("distances") else 0.0,
            }
        )
    return hits

def collection_stats() -> Dict:
    collection = get_collection()
    return {"total_chunks": collection.count()}

def reset_collection():
    """Wipe the collection (used for a clean-slate demo / debugging)."""
    try:
        _client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
