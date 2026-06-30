
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()  # reads GROQ_API_KEY (and any other vars) from a .env file in the project root

MODEL = "llama-3.1-8b-instant"

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY environment variable is not set. "
                "Get a free key at https://console.groq.com/keys"
            )
        _client = Groq(api_key=api_key)
    return _client


SYSTEM_PROMPT = """You are a document Q&A assistant. You will be given a user question
and a set of numbered source excerpts pulled from PDFs the user uploaded.

Rules:
1. Answer ONLY using information contained in the excerpts. Do not use outside knowledge.
2. If the excerpts do not contain enough information to answer, say so plainly instead of guessing.
3. Every factual claim in your answer must reference the excerpt number(s) it came from, like [1] or [2,3].
4. Be concise and direct. Do not repeat the excerpts verbatim at length; synthesize.
"""


def _build_context(chunks):
    lines = []
    for i, c in enumerate(chunks, start=1):
        meta = c["metadata"]
        lines.append(
            f"[{i}] (source: {meta['source']}, page {meta['page']})\n{c['text']}"
        )
    return "\n\n".join(lines)


def answer_question(question: str, chunks: list) -> dict:
    """
    Returns:
        {
            "answer": str,
            "citations": [{"index": int, "source": str, "page": int, "text": str}],
        }
    """
    if not chunks:
        return {
            "answer": (
                "I couldn't find anything in the uploaded documents relevant to "
                "that question. Try rephrasing, or confirm the right PDF was uploaded."
            ),
            "citations": [],
        }

    context = _build_context(chunks)
    user_prompt = f"Excerpts:\n\n{context}\n\nQuestion: {question}"

    client = _get_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=600,
    )
    answer_text = response.choices[0].message.content

    citations = [
        {
            "index": i + 1,
            "source": c["metadata"]["source"],
            "page": c["metadata"]["page"],
            "text": c["text"][:300] + ("..." if len(c["text"]) > 300 else ""),
        }
        for i, c in enumerate(chunks)
    ]

    return {"answer": answer_text, "citations": citations}
