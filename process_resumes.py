# --- process_resumes.py ---
import os
import faiss
import pickle
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
from PyPDF2 import PdfReader

faiss_folder = "faiss_store/"
os.makedirs(faiss_folder, exist_ok=True)
index_file = os.path.join(faiss_folder, "index.faiss")
metadata_file = os.path.join(faiss_folder, "metadata.pkl")

embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def process_and_store_resumes():
    resume_store = {}

    if os.path.exists(index_file) and os.path.exists(metadata_file):
        index = faiss.read_index(index_file)
        with open(metadata_file, "rb") as f:
            resume_store = pickle.load(f)
    else:
        index = faiss.IndexFlatIP(384)  # Cosine similarity (normalize vectors)

    resumes_folder = "resumes/"
    for resume_file in os.listdir(resumes_folder):
        resume_path = os.path.join(resumes_folder, resume_file)
        with open(resume_path, "rb") as f:
            reader = PdfReader(f)
            resume_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

        embedding = embed_model.embed_query(resume_text)
        vector = np.array(embedding).astype("float32").reshape(1, -1)
        vector /= np.linalg.norm(vector, axis=1, keepdims=True)  # Normalize
        index.add(vector)

        resume_store[len(resume_store)] = {"filename": resume_file, "text": resume_text}

    faiss.write_index(index, index_file)
    with open(metadata_file, "wb") as f:
        pickle.dump(resume_store, f)

    print("✅ Resumes processed & stored successfully!")