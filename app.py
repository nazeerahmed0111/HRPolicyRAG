from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import faiss
import os
from sentence_transformers import SentenceTransformer
import requests

app = FastAPI()

# Allow frontend to call the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load FAISS index and HR policy chunks
try:
    index = faiss.read_index("policy_index.index")
    with open("chunks.txt", "r", encoding="utf-8") as f:
        text_chunks = f.readlines()
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    raise RuntimeError("❌ Failed to load FAISS index or text chunks. Run prepare_policy.py.") from e

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(payload: QuestionRequest):
    question = payload.question.strip()

    if not question:
        return {"answer": "⚠️ Please provide a valid question."}

    try:
        q_embed = embedder.encode([question])
        _, I = index.search(q_embed, k=3)
        context = "\n".join([text_chunks[i] for i in I[0]])

        prompt = f"""You are an expert HR assistant. Use the following HR policy to answer clearly:

{context}

Question: {question}
Answer:"""

        response = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        })
        response.raise_for_status()
        result = response.json()
        answer = result.get("response", "").strip()

        return {"answer": answer or "⚠️ No response from model."}
    except Exception as e:
        print(f"Error: {e}")
        return {"answer": "⚠️ Error occurred while processing the question."}
