import streamlit as st
import requests
import fitz  # PyMuPDF
import re
import json
import os

# === API Keys (now loaded from environment variables) ===
API_KEY = os.getenv("GROQ_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
SERPER_KEY = os.getenv("SERPER_KEY")

# === Endpoints ===
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

# === Memory Functions ===
def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_memory(chat_history):
    with open(MEMORY_FILE, "w") as f:
        json.dump(chat_history, f, indent=2)

# === Wikipedia Helper ===
def get_wikipedia_summary(query):
    try:
        cleaned_query = re.sub(r'[^a-zA-Z0-9 ]', '', query).strip()
        search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={cleaned_query}&format=json"
        search_response = requests.get(search_url)
        results = search_response.json().get("query", {}).get("search", [])
        if not results:
            return None
        top_title = results[0]['title'].replace(" ", "_")
        summary_url = WIKIPEDIA_URL + top_title
        res = requests.get(summary_url)
        if res.status_code == 200:
            extract = res.json().get("extract", "")
            if extract:
                return extract

        fallback_title = cleaned_query.lower().replace(" ", "_")
        res_fallback = requests.get(WIKIPEDIA_URL + fallback_title)
        if res_fallback.status_code == 200:
            return res_fallback.json().get("extract", "")

    except Exception:
        pass
    return None

# === NewsAPI ===
def get_newsapi_headlines(query):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "pageSize": 5,
        "sortBy": "publishedAt",
        "apiKey": NEWSAPI_KEY
    }
    res = requests.get(url, params=params)
    if res.status_code != 200:
        return None
    articles = res.json().get("articles", [])
    if not articles:
        return None
    return "\n".join([f"- {a['title']} (📅 {a['publishedAt'][:10]}, Source: {a['source']['name']})" for a in articles])

# === Serper Fallback ===
def get_serper_results(query):
    headers = {"X-API-KEY": SERPER_KEY, "Content-Type": "application/json"}
    payload = {"q": query}
    res = requests.post(SERPER_URL, headers=headers, json=payload)
    try:
        results = res.json()
        items = results.get("organic", [])[:3]
        return "\n".join([f"- {r['title']}\n  {r['snippet']}\n  🔗 {r['link']}" for r in items])
    except Exception:
        return None

# === Translator ===
def translate_to_english(text):
    payload = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": "Translate non-English input to English. If already English, return it exactly as-is."},
            {"role": "user", "content": text}
        ],
        "temperature": 0
    }
    res = requests.post(GROQ_URL, headers=HEADERS, json=payload)
    try:
        return res.json()["choices"][0]["message"]["content"].strip()
    except:
        return text

# === Groq Document Q&A ===
def ask_groq(question, context):
    payload = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{context}\n\nQuestion: {question}"}
        ],
        "temperature": 0.4
    }
    res = requests.post(GROQ_URL, headers=HEADERS, json=payload)
    try:
        return res.json()["choices"][0]["message"]["content"]
    except:
        return "⚠️ No response"

# === Sidebar ===
st.sidebar.title("📂 Document Assistant")
section = st.sidebar.radio("Go to:", ["💬 Chatbot", "📄 Document Q&A"])

uploaded_file = st.sidebar.file_uploader("Upload .txt or .pdf", type=["txt", "pdf"])
document_text = ""
if uploaded_file:
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    ext = uploaded_file.name.split(".")[-1].lower()
    if ext == "txt":
        document_text = open(file_path, "r", encoding="utf-8").read()
    elif ext == "pdf":
        with fitz.open(file_path) as doc:
            document_text = "\n".join(page.get_text() for page in doc)
    st.sidebar.success(f"{uploaded_file.name} uploaded and saved!")

# === Chatbot UI ===
if section == "💬 Chatbot":
    st.title("🧠 AI Chatbot with Real-Time + Wikipedia Intelligence")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = load_memory()
        if not st.session_state.chat_history:
            st.session_state.chat_history = [{"role": "system", "content": "You are a helpful assistant with live and historical knowledge."}]

    for msg in st.session_state.chat_history[1:]:
        st.chat_message(msg["role"]).write(msg["content"])

    chat_input = st.chat_input("Ask me anything...")
    if chat_input:
        st.chat_message("user").write(chat_input)
        st.session_state.chat_history.append({"role": "user", "content": chat_input})

        translated = translate_to_english(chat_input)

        is_educational = any(k in translated.lower() for k in ["what is", "who is", "who was", "explain", "define", "history of", "origin of", "when did", "how did"])

        if is_educational:
            info = get_wikipedia_summary(translated)
        else:
            info = get_newsapi_headlines(translated) or get_serper_results(translated)

        context = info if info else "No relevant information found."

        prompt = (
            "IMPORTANT: Answer using ONLY the context provided below. Avoid internal knowledge unless necessary.\n\n"
            f"Context:\n{context}\n\nQuestion: {translated}"
        )

        payload = {
            "model": "llama3-70b-8192",
            "messages": [
                {"role": "system", "content": "You are an intelligent assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3
        }

        with st.spinner("Thinking..."):
            res = requests.post(GROQ_URL, headers=HEADERS, json=payload)
            try:
                reply = res.json()["choices"][0]["message"]["content"]
            except:
                reply = "⚠️ No response"

        st.chat_message("assistant").write(reply)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        save_memory(st.session_state.chat_history)

# === Document Q&A UI ===
if section == "📄 Document Q&A":
    st.title("📄 Ask Questions About Your Uploaded Document")

    if not document_text:
        st.warning("Please upload a document first.")
    else:
        q = st.text_input("What do you want to know?")
        if q:
            with st.spinner("Reading..."):
                a = ask_groq(q, document_text)
                st.subheader("💡 Answer:")
                st.write(a)
