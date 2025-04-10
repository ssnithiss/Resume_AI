import os
import faiss
import pickle
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings

def normalize_vector(vector):
    """Normalize vector for cosine similarity search in FAISS."""
    return vector / np.linalg.norm(vector, axis=1, keepdims=True)

# Paths
faiss_folder = "faiss_store/"
index_file = os.path.join(faiss_folder, "index.faiss")
metadata_file = os.path.join(faiss_folder, "metadata.pkl")

# Load embeddings
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_faiss_index_and_metadata():
    """Safely load FAISS index and metadata."""
    if os.path.exists(index_file) and os.path.exists(metadata_file):
        index = faiss.read_index(index_file)
        with open(metadata_file, "rb") as f:
            metadata = pickle.load(f)
        return index, metadata
    else:
        return None, None

def retrieve_resume_by_filename(filename):
    """Retrieve full resume text from metadata using filename."""
    _, metadata = load_faiss_index_and_metadata()
    if metadata:
        for entry in metadata.values():
            if entry["filename"] == filename:
                return entry["text"]
    return None

def retrieve_top_k_resumes(query, k=5):
    """Retrieve top-K resumes similar to job description using cosine similarity and return with scores."""
    index, metadata = load_faiss_index_and_metadata()
    if index is None or metadata is None:
        return []

    query_embedding = embed_model.embed_query(query)
    query_vector = np.array(query_embedding).astype("float32").reshape(1, -1)
    query_vector = normalize_vector(query_vector)

    scores, indices = index.search(query_vector, k)
    top_resumes = []
    
    for score, idx in zip(scores[0], indices[0]):
        if idx in metadata:
            resume_data = metadata[idx]
            resume_data["score"] = round(float(score), 4)
            top_resumes.append(resume_data)

    # Sort resumes by score in descending order (most relevant first)
    top_resumes = sorted(top_resumes, key=lambda x: x["score"], reverse=True)
    return top_resumes
