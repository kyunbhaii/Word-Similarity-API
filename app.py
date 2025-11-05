from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from pathlib import Path
import pickle

with open("word2vec_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI(
    title="Word Similarity API",
    description="Returns top N most similar words using a trained Word2Vec model",
    version="1.1.0",
)

class SimilarWord(BaseModel):
    word: str
    score: float

class SimilarResponse(BaseModel):
    query: str
    topn: int
    results: list[SimilarWord]

@app.get("/", summary="Home")
def home():
    return {"message": "Word Similarity API is running!"}

@app.get(
    "/similar/{word}",
    response_model=SimilarResponse,
    summary="Get Similar Words",
)
def get_similar_words(
    word: str,
    topn: int = Query(5, ge=1, le=25, description="Number of similar words to return (1–25)")
):
    
    query = (word or "").strip().lower()
    if not query:
        raise HTTPException(status_code=400, detail="Input word cannot be empty")

    if query not in model.wv:
        raise HTTPException(status_code=404, detail=f"'{query}' not found in vocabulary")

    raw = model.wv.most_similar(query, topn=topn)
    results = [SimilarWord(word=w, score=round(float(s), 4)) for w, s in raw]
    return SimilarResponse(query=query, topn=topn, results=results)

@app.get("/health", summary="Health")
def health():
    try:
        _ = next(iter(model.wv.key_to_index))
        return {"status": "ok", "model_loaded": True}
    except Exception as e:
        return {"status": "error", "model_loaded": False, "detail": str(e)}