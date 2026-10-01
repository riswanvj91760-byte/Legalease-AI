import os

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from google import genai


# ============================================================
# Configuration
# ============================================================

load_dotenv(".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY is not configured in the .env file.")
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="LegalEase AI",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ LegalEase AI")
st.caption("AI-Powered Legal Document Q&A")


# ============================================================
# PDF Reader
# ============================================================

def read_pdf(file):
    """Extract text from an uploaded PDF."""
    pdf = PdfReader(file)

    text = ""

    for page_number, page in enumerate(pdf.pages, start=1):
        try:
            page_text = page.extract_text() or ""
            text += f"\n\n--- Page {page_number} ---\n\n"
            text += page_text
        except Exception:
            continue

    return text.strip()


# ============================================================
# Text Chunking
# ============================================================

def chunk_text(text, chunk_size=1200):
    """
    Split document text into manageable chunks.

    chunk_size is measured approximately in words.
    """
    words = text.split()

    if not words:
        return []

    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]


# ============================================================
# Find relevant document chunks
# ============================================================

def find_relevant_chunks(chunks, query, max_chunks=3):
    """
    Simple keyword-based retrieval.

    This avoids requiring embeddings or PyTorch.
    """

    query_words = {
        word.lower().strip(".,!?;:()[]{}\"'")
        for word in query.split()
        if len(word.strip(".,!?;:()[]{}\"'")) > 2
    }

    scored_chunks = []

    for index, chunk in enumerate(chunks):
        chunk_words = set(
            word.lower().strip(".,!?;:()[]{}\"'")
            for word in chunk.split()
        )

        score = len(query_words.intersection(chunk_words))

        scored_chunks.append((score, index, chunk))

    scored_chunks.sort(
        key=lambda item: item[0],
        reverse=True
    )

    relevant = [
        item[2]
        for item in scored_chunks[:max_chunks]
        if item[0] > 0
    ]

    # If no keyword match is found, provide the first chunk
    # so Gemini can still attempt to answer.
    if not relevant and chunks:
        relevant = chunks[:1]

    return relevant


# ============================================================
# Gemini Legal Q&A
# ============================================================

def ask_gemini(question, context):
    """Ask Gemini a question using the document context."""

    prompt = f"""
You are LegalEase AI, an AI assistant for understanding legal documents.

Your task is to answer the user's question using ONLY the provided
document context.

IMPORTANT RULES:

1. Do not invent information.
2. If the answer cannot be found in the provided document,
   clearly say that the document does not provide enough information.
3. Explain legal terminology in simple language.
4. Do not present yourself as a lawyer.
5. Do not claim that your answer is legal advice.
6. Mention uncertainty when the document is unclear.
7. Keep the answer directly relevant to the user's question.
8. When possible, identify the relevant page or section from the
   supplied context.

DOCUMENT CONTEXT:
----------------
{context}
----------------

USER QUESTION:
{question}

Provide a clear, simple answer.
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    return response.text


# ============================================================
# File Upload
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a legal PDF",
    type=["pdf"],
)

if uploaded_file is not None:

    st.success("✅ PDF uploaded successfully.")

    # Extract text
    with st.spinner("Reading your document..."):
        document_text = read_pdf(uploaded_file)

    if not document_text:
        st.error(
            "Could not extract text from this PDF. "
            "The document may be scanned/image-based and require OCR."
        )
        st.stop()

    # Store document information
    st.session_state["document_text"] = document_text

    chunks = chunk_text(document_text)

    st.session_state["document_chunks"] = chunks

    # Document information
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Pages",
            len(PdfReader(uploaded_file).pages)
        )

    with col2:
        st.metric(
            "Text Chunks",
            len(chunks)
        )

    st.divider()

    # Preview
    with st.expander("📄 View extracted document text"):
        preview = document_text[:5000]

        st.text_area(
            "Extracted text",
            preview,
            height=300,
            disabled=True,
        )

    st.divider()

    # ========================================================
    # Question
    # ========================================================

    st.subheader("💬 Ask about your document")

    query = st.text_input(
        "Enter your question",
        placeholder="Example: What is the termination clause?",
    )

    if query:

        if not query.strip():
            st.warning("Please enter a question.")
            st.stop()

        relevant_chunks = find_relevant_chunks(
            chunks,
            query,
            max_chunks=3,
        )

        if not relevant_chunks:
            st.warning(
                "No relevant section was found in the document."
            )
            st.stop()

        context = "\n\n".join(relevant_chunks)

        with st.spinner("LegalEase AI is analyzing the document..."):

            try:

                answer = ask_gemini(
                    query,
                    context,
                )

                st.subheader("⚖️ Answer")

                st.write(answer)

                with st.expander("🔎 Document context used"):
                    st.write(context)

            except Exception as e:

                st.error(
                    "Gemini could not process the request."
                )

                st.code(str(e))


# ============================================================
# Empty state
# ============================================================

else:

    st.info(
        "👆 Upload a legal PDF to begin."
    )

    st.markdown(
        """
### What LegalEase AI can do

- 📄 Read legal PDF documents
- 🔍 Find relevant sections
- 💬 Answer questions about the document
- 🧠 Explain legal terminology in simpler language
- ⚖️ Provide document-based explanations
- 🚫 Avoid making up information that isn't present in the document
        """
    )