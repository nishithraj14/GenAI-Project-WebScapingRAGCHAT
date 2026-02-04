"""
Multi-PDF RAG Chatbot – Streamlit Demo

⚠️ DEMO ONLY
- Local: uses OPENAI_API_KEY from .env
- Streamlit Cloud: uses OPENAI_API_KEY from Secrets
- Core Flask + FAISS pipeline remains unchanged
"""

import os
import sys
import streamlit as st

# --------------------------------------------------
# Streamlit Page Config (MUST BE FIRST STREAMLIT CALL)
# --------------------------------------------------
st.set_page_config(
    page_title="Multi-PDF RAG Chatbot",
    page_icon="📄",
    layout="wide"
)

# --------------------------------------------------
# Ensure project root is in Python path
# --------------------------------------------------
sys.path.append(os.getcwd())

# --------------------------------------------------
# SAFE OpenAI API Key Handling (LOCAL + CLOUD)
# --------------------------------------------------
OPENAI_API_KEY = None

# 1️⃣ Try Streamlit Cloud secrets (SAFE)
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    OPENAI_API_KEY = None

# 2️⃣ Fallback to local environment (.env)
if not OPENAI_API_KEY:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 3️⃣ Hard stop if still missing
if not OPENAI_API_KEY:
    st.error(
        "OPENAI_API_KEY not configured.\n\n"
        "• Local: set it in .env\n"
        "• Streamlit Cloud: App → Settings → Secrets"
    )
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# --------------------------------------------------
# LangChain / FAISS imports (UNCHANGED backend)
# --------------------------------------------------
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain
)
from langchain.chains.combine_documents import create_stuff_documents_chain

from prompts import qa_system_prompt, contextualize_q_system_prompt

# --------------------------------------------------
# UI Header
# --------------------------------------------------
st.title("📄 Multi-PDF RAG Chatbot (Demo)")
st.caption(
    "Chat with multiple documents using Retrieval-Augmented Generation (RAG).\n"
    "Answers are strictly grounded in the PDFs."
)

st.divider()

# --------------------------------------------------
# Sidebar – PDF Browser
# --------------------------------------------------
st.sidebar.header("📚 Indexed Documents")

PDF_DIR = "static"
pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]

selected_pdf = st.sidebar.selectbox("Available PDFs", pdf_files)

if selected_pdf:
    with open(os.path.join(PDF_DIR, selected_pdf), "rb") as f:
        st.sidebar.download_button(
            "Download PDF",
            f,
            file_name=selected_pdf,
            mime="application/pdf"
        )

# --------------------------------------------------
# Example Questions
# --------------------------------------------------
st.markdown("### 💡 Example Questions")
st.markdown(
    """
- What is meant by **good faith**?
- Explain **strategic responsibility**
- Summarize **working intelligently**
- What does **solidarity oblige us** to do?
"""
)

st.divider()

# --------------------------------------------------
# Load Vector Store (cached)
# --------------------------------------------------
@st.cache_resource
def load_vectorstore():
    embeddings = OpenAIEmbeddings()
    faiss_path = "faiss"

    if os.path.exists(faiss_path):
        return FAISS.load_local(
            faiss_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

    loader = DirectoryLoader(
        PDF_DIR,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(faiss_path)
    return vectorstore

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever()

# --------------------------------------------------
# LLM + RAG Chain
# --------------------------------------------------
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualize_prompt
)

qa_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(
    history_aware_retriever,
    qa_chain
)

chat_store = {}

def get_history(session_id):
    if session_id not in chat_store:
        chat_store[session_id] = ChatMessageHistory()
    return chat_store[session_id]

conversational_chain = RunnableWithMessageHistory(
    rag_chain,
    get_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

# --------------------------------------------------
# Chat UI
# --------------------------------------------------
query = st.chat_input("Ask a question about the documents")

if query:
    with st.spinner("Searching documents..."):
        response = conversational_chain.invoke(
            {"input": query},
            config={"configurable": {"session_id": "demo"}}
        )

    st.chat_message("user").write(query)
    st.chat_message("assistant").write(response["answer"])

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()
st.caption(
    "Streamlit demo UI • Backend powered by LangChain + FAISS + OpenAI"
)
