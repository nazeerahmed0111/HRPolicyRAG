from flask import Flask, request, jsonify
from flask_cors import CORS
import faiss
import os
from sentence_transformers import SentenceTransformer
import requests

app = Flask(__name__)
CORS(app)  # Enable CORS so frontend can access this backend

# === Load the FAISS index and text chunks ===
try:
    index = faiss.read_index("policy_index.index")
    with open("chunks.txt", "r", encoding="utf-8") as f:
        text_chunks = f.readlines()
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    raise RuntimeError("❌ Failed to load FAISS index or text chunks. Run the text preparation script first.") from e

# === Ollama configuration ===
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"

# === API Route: POST /ask ===
@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"answer": "⚠️ Please provide a valid question."})

    try:
        # Step 1: Convert question to embedding
        q_embed = embedder.encode([question])

        # Step 2: Search for the top 3 most similar chunks
        _, I = index.search(q_embed, k=3)
        context = "\n".join([text_chunks[i] for i in I[0]])

        # Step 3: Prepare the prompt for Ollama
        prompt = f"""You are a highly knowledgeable and helpful HR assistant. Using the information provided from the company's HR policy document, answer the user's question accurately, clearly, and concisely. If the policy does not directly cover the question, reply based on general HR best practices while stating that the exact information wasn't found in the document.:

{context}

Question: {question}
Answer:"""

        # Step 4: Send the prompt to the local Ollama server
        response = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        })
        response.raise_for_status()
        result = response.json()

        # Step 5: Return the answer
        answer = result.get("response", "").strip()
        return jsonify({"answer": answer or "⚠️ No response from model."})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"answer": "⚠️ Error occurred while processing the question."})

# === Run the Flask app ===
if __name__ == "__main__":
    app.run(debug=True, port=8000)
