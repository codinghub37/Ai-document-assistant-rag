import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai



@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data
def create_embeddings(chunks):
    model = load_embedding_model()

    return model.encode(
        chunks,
        convert_to_numpy=True
    ).astype("float32")
# ===================================
# PAGE CONFIG
# ===================================

st.set_page_config(
    page_title="RAG Terms & Conditions Assistant",
    layout="wide"
)

# ===================================
# SIDEBAR
# ===================================

st.sidebar.title("Project Info")
st.sidebar.write("RAG-Based Terms & Conditions Assistant")
st.sidebar.write("Embedding Model: all-MiniLM-L6-v2")
st.sidebar.write("Vector Database: FAISS")
st.sidebar.write("LLM: Gemini 2.5 Flash")

api_key = st.sidebar.text_input(
    "Gemini API Key",
    type="password",
    placeholder="Enter Gemini API Key"
)

# ===================================
# CUSTOM CSS
# ===================================
st.markdown("""
<style>

/* Hide Streamlit Default UI */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* Main Background */
.stApp{
    background:#f1f5f9;
}

/* Remove Top Space */
.block-container{
    padding-top:0.8rem !important;
    max-width:1200px;
}

/* =========================
   SIDEBAR
========================= */

section[data-testid="stSidebar"]{
    background:linear-gradient(
        180deg,
        #334155 0%,
        #475569 100%
    );
    border-right:1px solid rgba(255,255,255,0.15);
}

section[data-testid="stSidebar"] *{
    color:white !important;
}

section[data-testid="stSidebar"] h1{
    color:white !important;
    font-size:40px !important;
    font-weight:700 !important;
}

section[data-testid="stSidebar"] p{
    font-size:16px !important;
    line-height:1.8 !important;
}

/* Sidebar Password Input */

section[data-testid="stSidebar"] [data-baseweb="input"]{
    background:#1e293b !important;
    border:2px solid #60a5fa !important;
    border-radius:12px !important;
}

section[data-testid="stSidebar"] .stTextInput input{
    background:#1e293b !important;
    color:white !important;
    border:none !important;
    border-radius:12px !important;
    padding:12px !important;
}

/* =========================
   TITLES
========================= */

h1{
    color:#0f172a !important;
    text-align:center;
    font-size:52px !important;
    font-weight:800 !important;
}

h2,h3{
    color:#0f172a !important;
}

/* Subtitle */

.subtitle{
    text-align:center;
    color:#475569;
    font-size:20px;
    margin-bottom:30px;
}

/* =========================
   LABELS
========================= */

label{
    color:#0f172a !important;
    font-weight:600 !important;
}

/* =========================
   FILE UPLOADER
========================= */

[data-testid="stFileUploader"]{
    background:white;
    border:2px dashed #2563eb;
    border-radius:18px;
    padding:20px;
    box-shadow:0 4px 12px rgba(0,0,0,0.08);
}

/* =========================
   INPUT BOX
========================= */

.stTextInput input{
    background:white !important;
    color:#0f172a !important;
    border:2px solid #2563eb !important;
    border-radius:12px !important;
    padding:12px !important;
}

.stTextInput input:hover{
    border-color:#1d4ed8 !important;
}

.stTextInput input:focus{
    border-color:#2563eb !important;
    box-shadow:0 0 0 3px rgba(37,99,235,0.15) !important;
}

/* =========================
   TEXT AREA
========================= */

.stTextArea textarea{
    background:white !important;
    color:#0f172a !important;
    border-radius:12px !important;
    border:1px solid #cbd5e1 !important;
}

.stTextArea textarea:focus{
    box-shadow:0 0 0 3px rgba(37,99,235,0.15) !important;
}

/* =========================
   SUCCESS BOX
========================= */

[data-testid="stAlert"]{
    background:#dbeafe !important;
    color:#1e3a8a !important;
    border:1px solid #93c5fd !important;
    border-radius:12px !important;
}

/* =========================
   EXPANDER
========================= */

details{
    background:white !important;
    border-radius:12px !important;
    padding:12px !important;
    border:1px solid #e2e8f0 !important;
}

/* =========================
   ANSWER TEXT
========================= */

p,
span,
div{
    color:#0f172a;
}

/* =========================
   CARDS LOOK
========================= */

[data-testid="stVerticalBlock"]{
    border-radius:18px;
}

/* =========================
   SCROLLBAR
========================= */

::-webkit-scrollbar{
    width:8px;
}

::-webkit-scrollbar-track{
    background:#e2e8f0;
}

::-webkit-scrollbar-thumb{
    background:#94a3b8;
    border-radius:10px;
}

::-webkit-scrollbar-thumb:hover{
    background:#64748b;
}

</style>
""", unsafe_allow_html=True)

# ===================================
# TITLE
# ===================================

st.markdown(
    "<h1 style='text-align:center;'>RAG Terms & Conditions Assistant</h1>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Upload any company Terms & Conditions PDF and ask questions using RAG + Gemini AI.
    </div>
    """,
    unsafe_allow_html=True
)

# ===================================
# FILE UPLOAD
# ===================================

uploaded_file = st.file_uploader(
    " Upload PDF Document",
    type=["pdf"]
)

# ===================================
# MAIN LOGIC
# ===================================

if uploaded_file:

    st.success(" PDF Uploaded Successfully")

    # Read PDF
    pdf_reader = PdfReader(uploaded_file)

    text = ""

    for page in pdf_reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    # Empty Check
    if not text.strip():

        st.error("No readable text found in PDF.")

        st.stop()

    # ===================================
    # PREVIEW
    # ===================================

    st.subheader(" Document Preview")

    st.text_area(
        "Extracted Text",
        text[:3000],
        height=250
    )

    # ===================================
    # CHUNKING
    # ===================================

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)

    st.success(
        f" Total Chunks Created: {len(chunks)}"
    )

    # ===================================
    # EMBEDDINGS
    # ===================================

    embed_model = load_embedding_model()
    embeddings = create_embeddings(chunks)

    st.success(
        f" Embeddings Created: {len(embeddings)}"
    )

    # ===================================
    # FAISS VECTOR DB
    # ===================================

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(embeddings)

    st.success(
        f" Vector Database Ready ({index.ntotal} vectors)"
    )

    # ===================================
    # GEMINI SETUP
    # ===================================

    if not api_key:

        st.warning(
            "Please enter Gemini API Key in sidebar."
        )

        st.stop()

    genai.configure(
        api_key=api_key
    )

    gemini_model = genai.GenerativeModel(
        "models/gemini-2.5-flash"
    )

    # ===================================
    # QUESTION INPUT
    # ===================================

    st.subheader(" Ask a Question")

    question = st.text_input(
    "Enter your question",
    placeholder="e.g. What is the refund policy?"
)

    if question:

        # Question Embedding

        question_embedding = embed_model.encode(
            [question],
            convert_to_numpy=True
        ).astype("float32")

        # Similarity Search

        distances, indices = index.search(
            question_embedding,
            k=3
        )

        context = ""

        for idx in indices[0]:

            if idx < len(chunks):

                context += chunks[idx]
                context += "\n\n"

        # Prompt

        prompt = f"""
You are a helpful AI assistant.

First try to answer using the document context.

If the answer exists in the document,
use only the document information.

If the answer does not exist in the document,
then answer using your general knowledge and clearly mention:

"Answer based on general knowledge, not the uploaded document."

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

        # Gemini Response
        

        try:

            with st.spinner("Generating answer..."):

                response = gemini_model.generate_content(
                    prompt
                )

            st.subheader("🤖 Answer")

            st.write(response.text)

        except Exception as e:

            st.error(
                f"Gemini Error: {str(e)}"
            )

        # Retrieved Chunks

        with st.expander("📚 View Retrieved Context"):

            st.write(context)

