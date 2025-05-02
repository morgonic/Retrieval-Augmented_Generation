import os
import logging
import warnings
import numpy as np
import faiss

from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import pipeline, logging as hf_logging

# --- Logging & Warning Suppression ---
logging.getLogger("langchain.text_splitter").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# --- Configuration Variables ---
chunk_size = 1000
chunk_overlap = 100
model_name = "sentence-transformers/all-distilroberta-v1"
top_k = 5

# --- Read the selected document ---
text = ""
file_path = "Selected_Document.txt"

if os.path.exists(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
else:
    print("⚠️ Selected_Document.txt not found. Run the scraper first.")
    exit(1)

if not text.strip():
    print("⚠️ The file is empty. Nothing to embed.")
    exit(1)

print("🔪 Splitting text into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_text(text)

if not chunks:
    print("⚠️ No chunks generated from text.")
    exit(1)

print(f"🧩 {len(chunks)} chunks created.")

print("🔍 Loading embedding model...")
embedding_model = SentenceTransformer(model_name)

print("🧠 Encoding chunks...")
embeddings = embedding_model.encode(chunks, show_progress_bar=True)
embeddings = np.array(embeddings, dtype=np.float32)

print("📚 Building FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("🤖 Loading FLAN-T5 generator...")
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    device=-1  # CPU
)

def retrieve_chunks(question, k=top_k):
    """
    Encode the question, search the FAISS index, and return the top k (chunk, distance) pairs.
    """
    query_vector = embedding_model.encode([question], show_progress_bar=False)
    query_vector = np.array(query_vector, dtype=np.float32)
    distances, indices = index.search(query_vector, k)

    results = []
    for j, i in enumerate(indices[0]):
        if i < len(chunks):
            results.append((chunks[i], distances[0][j]))
    return results


def answer_question(question, show_context=False):
    """
    Use retrieved chunks as context, build a prompt, and generate an answer.
    """
    retrieved = retrieve_chunks(question)
    if not retrieved:
        return "No relevant context found."

    if show_context:
        print("\n🔎 Top retrieved chunks:")
        for i, (chunk, dist) in enumerate(retrieved, 1):
            print(f"\n--- Chunk #{i} (Distance: {dist:.4f}) ---\n{chunk[:500]}{'...' if len(chunk) > 500 else ''}")

    context = "\n\n".join(chunk for chunk, _ in retrieved)
    prompt = f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {question}\nAnswer:"
    result = generator(prompt, max_length=256, do_sample=False)[0]["generated_text"]
    return result.strip()


if __name__ == "__main__":
    print("\n🧠 Retrieval-Augmented Generation App Ready!")
    print("Type your questions below. Type 'exit' or 'quit' to stop.\n")
    while True:
        question = input("Your question: ")
        if question.strip().lower() in ("exit", "quit"):
            print("👋 Goodbye!")
            break
        answer = answer_question(question)
        print(f"\n📎 Answer: {answer}\n")
