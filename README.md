# Resume_AI

Rag powered Resume AI web Application by ssnithiss with three modes

# 📄 Resume Screening RAG App (Gemini + FAISS + Streamlit)

A smart, interactive Resume Screening Web App built using **LLMs (Gemini API)**, **FAISS vector database**, and **Streamlit**. This app helps HR professionals to efficiently **screen**, **compare**, and **chat** with resumes based on job descriptions.

---

## 🚀 Features

### ✅ **Screening Mode**

- Upload multiple resumes (PDF).
- Enter a job description.
- Retrieve top-k relevant resumes using FAISS + semantic search.

### 💬 **Chat Mode**

- Select any uploaded resume.
- Chat with the content of the resume using Gemini API.
- Useful for deep Q&A-style review.

### ⚖️ **Comparison Mode**

- Compare two resumes side-by-side.
- Summarizes and compares:
  - 📌 Skills
  - 💼 Experience
  - 📚 Education
  - 🛠️ Technical Expertise
- Download PDF report of the comparison.

---

## 🛠️ Tech Stack

- **Frontend/UI**: Streamlit
- **LLM**: Google Gemini 1.5 Pro API
- **Vector DB**: FAISS
- **Frameworks**: LangChain, Python
- **PDF Handling**: PyMuPDF, ReportLab

---
