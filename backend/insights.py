import vector_store
import qa
def suggest_insights(max_chunks: int = 12) -> dict:
    collection = vector_store.get_collection()
    total = collection.count()
    if total == 0:
        return {"insights": "Upload some documents first to get suggested insights."}
    sample = collection.get(limit=min(max_chunks, total))
    chunks = [
        {"metadata": sample["metadatas"][i], "text": sample["documents"][i]}
        for i in range(len(sample["ids"]))
    ]
    client = qa._get_client()
    context = qa._build_context(chunks)
    prompt = (
        "Based on the document excerpts below, suggest:\n"
        "1) 3-5 key insights or themes across the documents\n"
        "2) 3 good follow-up questions the user could ask this system\n\n"
        f"Excerpts:\n\n{context}"
    )
    response = client.chat.completions.create(
        model=qa.MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful research assistant summarizing documents."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=500,
    )
    return {"insights": response.choices[0].message.content}
