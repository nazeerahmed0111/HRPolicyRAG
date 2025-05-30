import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import os

PDF_FILE = "D:\\VSCodes\\RAG_POC\\hr_policy.pdf"

def extract_text_chunks(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"{pdf_path} not found. Please add your HR policy PDF.")
    
    doc = fitz.open(pdf_path)
    chunks = []
    for page in doc:
        text = page.get_text()
        for paragraph in text.split("\n"):
            if paragraph.strip():
                chunks.append(paragraph.strip())
    return chunks

def main():
    print("🔍 Processing HR policy...")
    try:
        text_chunks = extract_text_chunks(PDF_FILE)
        if not text_chunks:
            raise ValueError("No text extracted from the document.")

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(text_chunks, show_progress_bar=True)

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        faiss.write_index(index, "policy_index.index")
        with open("chunks.txt", "w", encoding="utf-8") as f:
            for chunk in text_chunks:
                f.write(chunk + "\n")

        print("✅ Done! Index and chunks saved.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
