
import os
import re
from typing import List

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

import ingestor as ingest
import vector_store
import retrieval
import qa
import insights

UPLOAD_DIR = "./uploads"
MAX_FILES = 50
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200MB total upload cap

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="N-ERGY Document Intelligence")
templates = Jinja2Templates(directory="../frontend")


def secure_filename(filename: str) -> str:
    """Minimal stand-in for werkzeug's secure_filename: strips path parts
    and any characters that aren't alphanumeric, dot, dash, or underscore."""
    filename = os.path.basename(filename)
    filename = filename.strip().replace(" ", "_")
    filename = re.sub(r"(?u)[^-\w.]", "", filename)
    return filename or "file.pdf"


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/upload")
async def upload(files: List[UploadFile] = File(...)):
    if not files:
        return JSONResponse({"error": "No files received."}, status_code=400)
    if len(files) > MAX_FILES:
        return JSONResponse({"error": f"Max {MAX_FILES} files per upload."}, status_code=400)

    saved = []
    total_bytes = 0
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            continue
        filename = secure_filename(f.filename)
        path = os.path.join(UPLOAD_DIR, filename)
        content = await f.read()
        total_bytes += len(content)
        if total_bytes > MAX_UPLOAD_BYTES:
            return JSONResponse({"error": "Upload exceeds 200MB limit."}, status_code=400)
        with open(path, "wb") as out:
            out.write(content)
        saved.append((path, filename))

    if not saved:
        return JSONResponse({"error": "No valid PDF files found in upload."}, status_code=400)

    results = ingest.ingest_many(saved)

    summary = []
    total_chunks = 0
    for r in results:
        if r.error and r.is_empty:
            summary.append({"filename": r.filename, "status": "empty", "message": r.error})
            continue
        if r.error:
            summary.append({"filename": r.filename, "status": "error", "message": r.error})
            continue
        n = vector_store.add_chunks(r.chunks)
        total_chunks += n
        summary.append({"filename": r.filename, "status": "ok", "chunks": n})

    return {"files": summary, "total_chunks_added": total_chunks}


@app.post("/api/query")
async def query(request: Request):
    data = await request.json()
    question = (data or {}).get("question", "").strip()
    if not question:
        return JSONResponse({"error": "Question is required."}, status_code=400)

    if not retrieval.has_documents():
        return JSONResponse({"error": "No documents have been uploaded yet."}, status_code=400)

    chunks = retrieval.retrieve(question)
    result = qa.answer_question(question, chunks)
    return result


@app.get("/api/insights")
def get_insights():
    if not retrieval.has_documents():
        return JSONResponse({"error": "No documents have been uploaded yet."}, status_code=400)
    return insights.suggest_insights()


@app.get("/api/status")
def status():
    return vector_store.collection_stats()


@app.post("/api/reset")
def reset():
    vector_store.reset_collection()
    for fname in os.listdir(UPLOAD_DIR):
        try:
            os.remove(os.path.join(UPLOAD_DIR, fname))
        except OSError:
            pass
    return {"status": "reset"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5050, reload=True)
