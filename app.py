import os
from dotenv import load_dotenv
from flask import Flask, render_template, request

# =========================
# Environment Configuration
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not loaded from .env")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# =========================
# LangChain Imports (Modern)
# =========================
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from prompts import qa_system_prompt, contextualize_q_system_prompt

# =========================
# App Initialization
# =========================
app = Flask(__name__)

FAISS_PATH = "faiss"
conversation_store = {}

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# =========================
# Session History
# =========================
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in conversation_store:
        conversation_store[session_id] = ChatMessageHistory()
    return conversation_store[session_id]

# =========================
# Document Processing
# =========================
def load_documents():
    loader = DirectoryLoader(
        "static",
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    return loader.load()

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)

def get_vectorstore():
    path = os.path.join(BASE_DIR, FAISS_PATH)

    if os.path.exists(path):
        return FAISS.load_local(
            path,
            OpenAIEmbeddings(),
            allow_dangerous_deserialization=True
        )

    documents = load_documents()
    chunks = split_documents(documents)

    vectorstore = FAISS.from_documents(
        chunks,
        OpenAIEmbeddings()
    )
    vectorstore.save_local(path)
    return vectorstore

def get_retriever():
    return get_vectorstore().as_retriever()

# =========================
# Routes
# =========================
@app.route("/")
def index():
    return render_template("home.html")

@app.route("/chat", methods=["GET", "POST"])
def chat():
    if request.method == "GET":
        return render_template("chat.html")

    question = request.form["question"]

    retriever = get_retriever()

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        qa_chain
    )

    conversational_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    response = conversational_chain.invoke(
        {"input": question},
        config={"configurable": {"session_id": "default"}}
    )

    return render_template(
        "chat.html",
        chat_history=[question, response["answer"]]
    )

# =========================
# App Entry Point
# =========================
if __name__ == "__main__":
    app.run(debug=True)
