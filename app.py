import streamlit as st
import requests
import fitz  # PyMuPDF
import re
import json
import os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
SERPER_KEY = os.getenv("SERPER_KEY")

with st.sidebar.expander("🔐 API Key Status", expanded=True):
    st.text(f"GROQ_API_KEY loaded: {'✅' if API_KEY else '❌'}")
    st.text(f"NEWSAPI_KEY loaded: {'✅' if NEWSAPI_KEY else '❌'}")
    st.text(f"SERPER_KEY loaded: {'✅' if SERPER_KEY else '❌'}")

st.sidebar.write("📂 Working Directory:", os.getcwd())

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
SERPER_URL = "https://google.serper.dev/search"
WIKIPEDIA_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

MAX_DOC_CHARS = 5000
MEMORY_FILE = "chat_memory.json"
UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_memory(chat_history):
    with open(MEMORY_FILE, "w") as f:
        json.dump(chat_history, f, indent=2)

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

def get_serper_results(query):
    try:
        headers = {"X-API-KEY": SERPER_KEY, "Content-Type": "application/json"}
        payload = {"q": query}
        res = requests.post(SERPER_URL, headers=headers, json=payload)
        items = res.json().get("organic", [])[:3]
        return "\n".join([f"- {r['title']}\n  {r['snippet']}\n  🔗 {r['link']}" for r in items])
    except:
        return None

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

def get_all_documents():
    context = ""
    for file in os.listdir(UPLOAD_DIR):
        path = os.path.join(UPLOAD_DIR, file)
        if file.endswith(".txt"):
            context += open(path, "r", encoding="utf-8").read() + "\n"
        elif file.endswith(".pdf"):
            with fitz.open(path) as doc:
                context += "\n".join(page.get_text() for page in doc) + "\n"
    return context

def get_document_context(filename):
    path = os.path.join(UPLOAD_DIR, filename)
    if filename.endswith(".txt"):
        return open(path, "r", encoding="utf-8").read()
    elif filename.endswith(".pdf"):
        with fitz.open(path) as doc:
            return "\n".join(page.get_text() for page in doc)
    return ""

st.sidebar.title("📂 Document Assistant")
section = st.sidebar.radio("Go to:", ["💬 Chatbot", "📄 Document Q&A"])

if st.sidebar.button("🧹 Clear Chat History"):
    st.session_state.chat_history = []
    save_memory([])
    st.rerun()

uploaded_file = st.sidebar.file_uploader("Upload .txt or .pdf", type=["txt", "pdf"])
if uploaded_file:
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    if not os.path.exists(file_path):
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success(f"✅ Saved: {uploaded_file.name}")
        st.rerun()

if st.sidebar.button("📤 Push to GitHub"):
    os.system("git add uploaded_docs/*")
    os.system("git commit -m 'Add uploaded document(s)'")
    os.system("git push origin main")
    st.sidebar.success("✅ Pushed uploaded documents to GitHub")

if section == "📄 Document Q&A":
    st.title("📄 Ask Questions About Your Uploaded Document")

    doc_files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith((".txt", ".pdf"))]
    selected = st.selectbox("📑 Select document to query:", ["All"] + doc_files, index=0)

    question = st.text_input("What do you want to know?")
    if question:
        with st.spinner("Reading documents..."):
            translated = translate_to_english(question)
            if selected == "All":
                all_docs = {f: get_document_context(f) for f in doc_files}
                matching_docs = [text for name, text in all_docs.items() if re.search(translated[:40], text, re.IGNORECASE)]
                context = "\n\n".join(matching_docs)
            else:
                context = get_document_context(selected)
            answer = ask_groq(translated, context)
            st.subheader("💡 Answer:")
            st.write(answer)

if section == "💬 Chatbot":
    st.title("🧠 AI Chatbot with Smart Context")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = load_memory()
        if not st.session_state.chat_history:
            st.session_state.chat_history = [
                {"role": "system", "content": "You are a smart AI assistant with access to documents, Wikipedia, and real-time news. You handle questions like who, what, when, how many, how long, tell me how, tell me why, etc."}
            ]

    for msg in st.session_state.chat_history[1:]:
        st.chat_message(msg["role"]).write(msg["content"])

    chat_input = st.chat_input("Ask your assistant...")
    if chat_input:
        st.chat_message("user").write(chat_input)
        st.session_state.chat_history.append({"role": "user", "content": chat_input})

        translated = translate_to_english(chat_input)

        wiki = get_wikipedia_summary(translated)
        news = get_newsapi_headlines(translated)
        serper = get_serper_results(translated)
        docs = get_all_documents()

        combined_context = ""
        if wiki:
            combined_context += f"📚 Wikipedia:\n{wiki}\n\n"
        if news:
            combined_context += f"📡 NewsAPI:\n{news}\n\n"
        if serper:
            combined_context += f"🌐 Serper:\n{serper}\n\n"
        if translated.lower() in docs.lower():
            combined_context += f"📄 Document match:\n{docs}"

        answer = ask_groq(translated, combined_context.strip())

        st.chat_message("assistant").write(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        save_memory(st.session_state.chat_history)