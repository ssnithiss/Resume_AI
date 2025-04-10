import os
import faiss
import pickle
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Paths
faiss_folder = "faiss_store/"
index_file = os.path.join(faiss_folder, "index.faiss")
metadata_file = os.path.join(faiss_folder, "metadata.pkl")

def load_vectorstore():
    if not os.path.exists(index_file) or not os.path.exists(metadata_file):
        return None

    with open(metadata_file, "rb") as f:
        metadata = pickle.load(f)

    index = faiss.read_index(index_file)

    documents = []
    docstore_dict = {}
    index_to_docstore_id = {}

    for i, entry in enumerate(metadata.values()):
        doc = Document(page_content=entry["text"], metadata={"filename": entry["filename"]})
        docstore_dict[str(i)] = doc
        index_to_docstore_id[i] = str(i)
        documents.append(doc)

    docstore = InMemoryDocstore(docstore_dict)

    vectorstore = FAISS(
        embedding_model,
        index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )
    return vectorstore

def get_langchain_retriever(k=5):
    vectorstore = load_vectorstore()
    if vectorstore is None:
        return None
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})

def retrieve_resume_by_filename(filename):
    if not os.path.exists(metadata_file):
        return None
    with open(metadata_file, "rb") as f:
        metadata = pickle.load(f)
        for entry in metadata.values():
            if entry["filename"] == filename:
                return entry["text"]
    return None

def retrieve_top_k_resumes(query, k=5):
    retriever = get_langchain_retriever(k=k)
    if retriever is None:
        return []
    docs = retriever.get_relevant_documents(query)
    results = []
    for doc in docs:
        results.append({
            "filename": doc.metadata.get("filename", "Unknown"),
            "text": doc.page_content
        })
    return results
