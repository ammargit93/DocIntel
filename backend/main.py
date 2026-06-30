import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import ingestor as ingest
import vector_store
import retrieval
import qa
import insights
UPLOAD_DIR = "./uploads"
MAX_FILES = 50
os.makedirs(UPLOAD_DIR, exist_ok=True)
app = Flask(__name__, template_folder="../frontend", static_folder="../frontend")
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200MB total upload cap
# GET  / -> UI
@app.route("/")
def index():
    return render_template("index.html")
# POST /api/upload -> upload + ingest + embed N PDFs
@app.route("/api/upload", methods=["POST"])
def upload():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files received."}), 400
    if len(files) > MAX_FILES:
        return jsonify({"error": f"Max {MAX_FILES} files per upload."}), 400
    saved = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            continue
        filename = secure_filename(f.filename)
        path = os.path.join(UPLOAD_DIR, filename)
        f.save(path)
        saved.append((path, filename))
    if not saved:
        return jsonify({"error": "No valid PDF files found in upload."}), 400
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
    return jsonify({"files": summary, "total_chunks_added": total_chunks})
#  POST /api/query -> ask a question, get answer + citations
@app.route("/api/query", methods=["POST"])
def query():
    data = request.get_json(force=True)
    question = (data or {}).get("question", "").strip()
    if not question:
        return jsonify({"error": "Question is required."}), 400
    if not retrieval.has_documents():
        return jsonify({"error": "No documents have been uploaded yet."}), 400
    chunks = retrieval.retrieve(question)
    result = qa.answer_question(question, chunks)
    return jsonify(result)
#     GET  /api/insights -> suggested insights/next steps this is the additional
@app.route("/api/insights", methods=["GET"])
def get_insights():
    if not retrieval.has_documents():
        return jsonify({"error": "No documents have been uploaded yet."}), 400
    return jsonify(insights.suggest_insights())
# GET  /api/status -> how many chunks/docs are indexed
@app.route("/api/status", methods=["GET"])
def status():
    return jsonify(vector_store.collection_stats())
# POST /api/reset -> clear the index (for demo convenience)
@app.route("/api/reset", methods=["POST"])
def reset():
    vector_store.reset_collection()
    for fname in os.listdir(UPLOAD_DIR):
        try:
            os.remove(os.path.join(UPLOAD_DIR, fname))
        except OSError:
            pass
    return jsonify({"status": "reset"})
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)

