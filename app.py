import streamlit as st
import requests
import fitz  # PyMuPDF
import re
import json
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
SERPER_KEY = os.getenv("SERPER_KEY")
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
SERPER_URL = "https://google.serper.dev/search"
WIKIPEDIA_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

UPLOAD_DIR = "uploaded_docs"
VECTOR_DIR = "vector_db"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

MAX_DOC_CHARS = 5000
MEMORY_FILE = "chat_memory.json"

# Load or save chat history
def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_memory(chat_history):
    with open(MEMORY_FILE, "w") as f:
        json.dump(chat_history, f, indent=2)

# Embed all documents once and store in vector DB
def embed_documents():
    docs = []
    for file in os.listdir(UPLOAD_DIR):
        path = os.path.join(UPLOAD_DIR, file)
        if file.endswith(".txt"):
            loader = TextLoader(path)
        elif file.endswith(".pdf"):
            loader = PyPDFLoader(path)
        else:
            continue
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    vectordb = Chroma.from_documents(chunks, EMBEDDING_MODEL, persist_directory=VECTOR_DIR)
    vectordb.persist()

# Search vectors
def search_similar_docs(query, selected=None):
    vectordb = Chroma(persist_directory=VECTOR_DIR, embedding_function=EMBEDDING_MODEL)
    if selected == "All" or not selected:
        docs = vectordb.similarity_search(query, k=4)
    else:
        all_docs = vectordb.similarity_search(query, k=10)
        docs = [doc for doc in all_docs if selected.lower() in doc.metadata.get("source", "").lower()][:3]
    return "\n---\n".join([doc.page_content for doc in docs])

# Wikipedia

def get_wikipedia_summary(query):
    try:
        cleaned_query = re.sub(r'[^a-zA-Z0-9 ]', '', query).strip()
        search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={cleaned_query}&format=json"
        results = requests.get(search_url).json().get("query", {}).get("search", [])
        if not results:
            return None
        top_title = results[0]['title'].replace(" ", "_")
        summary_url = WIKIPEDIA_URL + top_title
        res = requests.get(summary_url)
        return res.json().get("extract", "") if res.status_code == 200 else None
    except:
        return None

# NewsAPI

def get_newsapi_headlines(query):
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "language": "en",
            "pageSize": 5,
            "sortBy": "publishedAt",
            "apiKey": NEWSAPI_KEY
        }
        res = requests.get(url, params=params)
        articles = res.json().get("articles", []) if res.status_code == 200 else []
        return "\n".join([f"- {a['title']} ({a['source']['name']}, {a['publishedAt'][:10]})" for a in articles]) if articles else None
    except:
        return None

# Serper

def get_serper_results(query):
    try:
        headers = {"X-API-KEY": SERPER_KEY, "Content-Type": "application/json"}
        payload = {"q": query}
        res = requests.post(SERPER_URL, headers=headers, json=payload)
        items = res.json().get("organic", [])[:3]
        return "\n".join([f"- {r['title']}\n  {r['snippet']}\n  🔗 {r['link']}" for r in items])
    except:
        return None

# Translate

def translate_to_english(text):
    try:
        payload = {
            "model": "llama3-70b-8192",
            "messages": [
                {"role": "system", "content": "Translate non-English input to English. If already English, return it exactly as-is."},
                {"role": "user", "content": text}
            ],
            "temperature": 0
        }
        res = requests.post(GROQ_URL, headers=HEADERS, json=payload)
        return res.json()["choices"][0]["message"]["content"].strip()
    except:
        return text

# Ask Groq

def ask_groq(question, context=None):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers based on relevant documents, Wikipedia, and news sources. Do not mix unrelated sources."}
    ]
    if context:
        messages.append({"role": "user", "content": f"Context:\n{context[:MAX_DOC_CHARS]}\n\nQuestion: {question}"})
    else:
        messages.append({"role": "user", "content": question})
    payload = {
        "model": "llama3-70b-8192",
        "messages": messages,
        "temperature": 0.3
    }
    try:
        res = requests.post(GROQ_URL, headers=HEADERS, json=payload)
        return res.json()["choices"][0]["message"]["content"]
    except:
        return "⚠️ No response"

# UI setup
st.sidebar.title("🧠 AI Toolbox")
section = st.sidebar.radio("Select Mode", ["💬 Chatbot", "📄 Documents Q&A"])

# Upload doc
uploaded_file = st.sidebar.file_uploader("Upload .txt or .pdf", type=["txt", "pdf"])
if uploaded_file:
    with open(os.path.join(UPLOAD_DIR, uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    embed_documents()
    st.sidebar.success("File uploaded and indexed.")
    st.rerun()

# === DOCUMENTS Q&A ===
if section == "📄 Documents Q&A":
    st.title("📄 Ask About Documents")
    all_files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith((".txt", ".pdf"))]
    selected_doc = st.selectbox("Choose document to search in", ["All"] + all_files)

    question = st.text_input("What do you want to know?")
    if question:
        with st.spinner("Searching documents..."):
            translated = translate_to_english(question)
            doc_context = search_similar_docs(translated, selected=selected_doc)
            response = ask_groq(translated, doc_context)
            st.subheader("💡 Answer:")
            st.write(response)

# === SMART CHATBOT ===
if section == "💬 Chatbot":
    st.title("🤖 Smart Chatbot with Web + Wiki + Docs")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = load_memory()
        if not st.session_state.chat_history:
            st.session_state.chat_history = [
                {"role": "system", "content": "You are a smart assistant using documents, Wikipedia, and news sources."}
            ]

    for msg in st.session_state.chat_history[1:]:
        st.chat_message(msg["role"]).write(msg["content"])

    chat_input = st.chat_input("Ask me anything...")
    if chat_input:
        st.chat_message("user").write(chat_input)
        st.session_state.chat_history.append({"role": "user", "content": chat_input})

        translated = translate_to_english(chat_input)
        wiki = get_wikipedia_summary(translated)
        news = get_newsapi_headlines(translated)
        serper = get_serper_results(translated)
        doc_context = search_similar_docs(translated)

        combined = ""
        if wiki: combined += f"📚 Wikipedia:\n{wiki}\n\n"
        if news: combined += f"📰 News:\n{news}\n\n"
        if serper: combined += f"🌐 Web:\n{serper}\n\n"
        if doc_context: combined += f"📄 Documents:\n{doc_context}"

        answer = ask_groq(translated, combined.strip())
        st.chat_message("assistant").write(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        save_memory(st.session_state.chat_history)
