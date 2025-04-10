import os
import shutil
import streamlit as st
from io import BytesIO
from reportlab.pdfgen import canvas
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage
from retrieve_resumes import retrieve_resume_by_filename, retrieve_top_k_resumes, get_langchain_retriever
from process_resumes import process_and_store_resumes

    
# Load API Key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GEMINI_API_KEY, temperature=0.2)

st.set_page_config(page_title="Resume Screening App", layout="wide")

# --- Sidebar ---
st.sidebar.header("📤 Upload & Process Resumes")
uploaded_files = st.sidebar.file_uploader("Upload Resume PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    os.makedirs("resumes", exist_ok=True)
    for file in uploaded_files:
        with open(os.path.join("resumes", file.name), "wb") as f:
            f.write(file.getbuffer())
    st.sidebar.success("✅ Resumes uploaded!")

if st.sidebar.button("🔄 Process Resumes"):
    process_and_store_resumes()
    st.sidebar.success("✅ Processed and stored in FAISS!")

if st.sidebar.button("🗑️ Delete All Data"):
    try:
        shutil.rmtree("resumes")
        os.remove("faiss_store/index.faiss")
        os.remove("faiss_store/metadata.pkl")
        st.sidebar.success("✅ All data deleted!")
    except Exception as e:
        st.sidebar.error(f"Error deleting: {str(e)}")

st.markdown("<h1 style='text-align: center;'>📂 Retrieval Augmented Generation powered Resume AI</h1>", unsafe_allow_html=True)

mode = st.selectbox("Choose a Mode", ["Screening Mode", "Chat Mode", "Comparison Mode"])

# --- Screening Mode ---
if mode == "Screening Mode":
    st.subheader("🧐 Screening Resumes by Job Description")
    job_description = st.text_area("Paste Job Description")
    k = st.slider("Select number of resumes to retrieve", 1, 10, 5)

    if st.button("🔍 Retrieve Matching Resumes"):
        if not job_description:
            st.warning("Please enter a job description.")
        else:
            top_resumes = retrieve_top_k_resumes(job_description, k)
            if not top_resumes:
                st.warning("No resumes found. Upload and process first.")
            else:
                st.subheader("Top Matching Resumes")
                for resume in top_resumes:
                    st.markdown(f"**📄 {resume['filename']}**")

# --- chat mode ---
elif mode == "Chat Mode":
    st.subheader("💬 Chat with a Resume")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "selected_resume" not in st.session_state:
        st.session_state.selected_resume = ""

    if not st.session_state.selected_resume:
        st.session_state.selected_resume = st.text_input("Enter Resume Filename to start chat:")

    if st.session_state.selected_resume:
        resume_text = retrieve_resume_by_filename(st.session_state.selected_resume)
        if not resume_text:
            st.error("Resume not found. Please check the filename.")
        else:
            st.markdown("**Start chatting below:**")
            user_input = st.chat_input("Type your message...")

            if user_input:
                # Construct prompt with resume context and question
                prompt = f"Given the following resume:\n\n{resume_text}\n\nAnswer this question:\n{user_input}"

                model = genai.GenerativeModel("gemini-1.5-pro")
                response = model.generate_content(prompt)
                answer = response.text.strip()

                # Save to chat history
                st.session_state.chat_history.append(("You", user_input))
                st.session_state.chat_history.append(("Bot", answer))

            # Display chat history
            for sender, message in st.session_state.chat_history:
                with st.chat_message(sender):
                    st.markdown(message)

            # Add reset button
            if st.button("🔁 Reset Chat"):
                st.session_state.chat_history = []
                st.session_state.selected_resume = ""
                
# --- Comparison Mode ---
elif mode == "Comparison Mode":
    st.subheader("🔍 Compare Two Resumes")
    resume1 = st.text_input("Enter first resume filename")
    resume2 = st.text_input("Enter second resume filename")

    if st.button("⚖️ Compare Resumes"):
        text1 = retrieve_resume_by_filename(resume1)
        text2 = retrieve_resume_by_filename(resume2)

        if not text1 or not text2:
            st.error("One or both resume files not found.")
        else:
            prompt = f"""
Compare the following two resumes.

Resume 1:
{text1}

Resume 2:
{text2}

Summarize and compare:
- Total experience
- Skills
- Projects
- Technical expertise
- Education
"""
            # Use the Gemini LLM wrapper here too
            comparison_response = llm.invoke(prompt)

            st.subheader("📊 Comparison Report")
            st.write(comparison_response.content)

            pdf_buffer = BytesIO()
            pdf = canvas.Canvas(pdf_buffer)
            pdf.setFont("Helvetica", 12)
            y = 800
            for line in comparison_response.content.split("\n"):
                pdf.drawString(30, y, line)
                y -= 20
            pdf.save()
            pdf_buffer.seek(0)
            st.download_button(
                label="📥 Download Comparison Report (PDF)",
                data=pdf_buffer,
                file_name="resume_comparison.pdf",
                mime="application/pdf",
            )
