# DocIntel — PDF Q&A with Citations
 
A document intelligence system: upload PDFs, ask questions in plain English, and get answers grounded in the documents with exact source/page citations, plus auto-generated insights across the uploaded set.
 
## Architecture Overview
 
```
Browser UI (frontend/index.html)
        │  upload PDFs / ask question
        ▼
FastAPI backend (backend/main.py)
        │
        ├── ingestor.py     → PDF text extraction (pypdf) + sentence-aware chunking
        ├── vector_store.py → embeds chunks (sentence-transformers, all-MiniLM-L6-v2)
        │                     and stores/searches them in ChromaDB (local, on disk)
        ├── retrieval.py    → top-k similarity search + distance-threshold filtering
        │                     (drops irrelevant chunks instead of forcing an answer)
        ├── qa.py           → builds a grounded prompt from retrieved chunks,
        │                     calls Groq (llama-3.1-8b-instant) for the answer,
        │                     maps citations back to source file + page
        └── insights.py     → samples stored chunks, asks Groq for themes/
                              follow-up questions across all uploaded docs
```
 
Each stage is a separate file so the pipeline is easy to follow and swap out (e.g. replacing ChromaDB doesn't touch ingestion or generation logic).
 
## Key Design Decisions
 
- **Embeddings**: `all-MiniLM-L6-v2` via sentence-transformers, run locally. No network call per chunk, fast on CPU, no API cost — good fit for the latency goal below.
- **Vector store**: ChromaDB with local disk persistence. Zero infrastructure to stand up and fast enough for the assignment's scale (≤50 PDFs). If I had more time, I'd move this to a hosted **pgvector** instance (e.g. via Supabase) instead of local disk — that would let the index survive across deployments/instances and support multiple concurrent users safely, rather than being tied to one machine's filesystem.
- **Retrieval scope**: right now every query searches across *all* PDFs ever uploaded, old and new — there's no "only search the latest batch" mode. The intended fix (not yet built): clear the previous collection/vectors before indexing a new batch, or add per-document filtering so a query can be scoped to just the documents currently relevant, instead of the whole history.
- **Citations are structural, not generated**: the LLM is given numbered excerpts and references them as `[1]`, `[2]`, etc.; the actual filename/page/text for each citation comes from stored chunk metadata, so the model can't invent a source that doesn't exist.
## Tradeoffs
 
I optimized for **latency over accuracy**. A larger embedding model (e.g. OpenAI's `text-embedding-3-small/large`) or a bigger LLM would likely produce more precise answers, but both add real network latency and cost. Groq was the deciding factor here — it serves open-source Llama models on custom hardware at very high tokens/sec, is free to use, and returns answers fast enough that the system feels responsive rather than something the user has to wait on. In practice, most users want a quick, "good enough" answer over a slower, marginally more accurate one — especially for an interactive Q&A tool rather than a one-shot research report. The cost is that retrieval and generation are both running on smaller/lighter models, so edge-case nuance can occasionally be missed.
 
## What Would Break at Scale (10k+ documents)
 
- **Local ChromaDB on disk** wasn't built for this volume or for concurrent access from multiple app instances — a managed vector DB (pgvector, Pinecone, Weaviate) would be needed.
- **Synchronous upload handling** — ingesting 10k PDFs in the request thread would time out; this needs a background job queue so uploads return immediately and indexing happens async.
- **No scoping/filtering at query time** — searching the entire corpus on every question gets slower and less precise as the document count grows; metadata filtering (by upload batch, folder, date) becomes necessary.
- **CPU-bound local embeddings** — fine for dozens of PDFs, but would bottleneck badly on bulk ingestion at this scale; batched GPU inference or a hosted embeddings API would be needed.
- **No re-ranking step** — pure similarity search loses precision as the corpus grows; a cross-encoder re-ranker on top candidates would help maintain answer quality.
## What I'd Improve With More Time
 
- Move the vector store from local ChromaDB to a hosted **Supabase/pgvector** instance so the index isn't tied to a single machine.
- Add a way to clear out previous vectors before indexing a new upload batch (or proper per-document filtering), so a question can be scoped to only the documents currently relevant instead of the entire upload history.
- Stream LLM responses token-by-token instead of waiting for the full completion.
- Add a re-ranking stage to recover accuracy traded away for speed.