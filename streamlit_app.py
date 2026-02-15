import streamlit as st
import requests
import json

API_URL = "http://localhost:8000/chat"
STREAM_URL = "http://localhost:8000/chat/stream"
UPLOAD_URL = "http://localhost:8000/upload"
REBUILD_URL = "http://localhost:8000/rebuild"
DOCS_URL = "http://localhost:8000/documents"

st.set_page_config(
    page_title="Enterprise RAG Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================
# THEME — inject custom CSS
# ==========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Variables ─────────────────────────────────────── */
:root {
  --cream:        #F7F4EE;
  --warm-white:   #FDFCF9;
  --sand:         #E8E2D6;
  --tan:          #C9BFA8;
  --bark:         #7A6A52;
  --ink:          #1C1713;
  --ink-60:       #6B6055;
  --ink-30:       #A89E93;
  --accent:       #C4531A;
  --accent-light: #F4E8DF;
  --accent-muted: #E8845A;
  --green:        #2D6A4F;
  --green-light:  #D8EDDF;
  --blue:         #1A4B8C;
  --blue-light:   #DCE8F8;
  --radius:       10px;
  --radius-sm:    6px;
  --shadow-sm:    0 1px 3px rgba(28,23,19,.06), 0 1px 2px rgba(28,23,19,.04);
  --shadow-md:    0 4px 12px rgba(28,23,19,.08), 0 2px 6px rgba(28,23,19,.04);
  --tr:           0.2s cubic-bezier(.4,0,.2,1);
}

/* ── Global reset ───────────────────────────────────── */
html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif !important;
  color: var(--ink) !important;
}

/* App background */
.stApp {
  background: var(--cream) !important;
}

/* ── Sidebar ────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: var(--warm-white) !important;
  border-right: 1.5px solid var(--sand) !important;
}

[data-testid="stSidebar"] > div:first-child {
  padding-top: 1.4rem;
}

/* Sidebar header text */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
  font-family: 'Libre Baskerville', serif !important;
  font-size: 0.85rem !important;
  font-weight: 700 !important;
  color: var(--ink) !important;
  letter-spacing: -0.01em;
  border-bottom: 1px solid var(--sand);
  padding-bottom: 6px;
  margin-bottom: 10px;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p {
  font-size: 0.82rem !important;
  color: var(--ink-60) !important;
}

/* Sidebar divider */
[data-testid="stSidebar"] hr {
  border-color: var(--sand) !important;
  margin: 12px 0 !important;
}

/* ── Buttons ────────────────────────────────────────── */
.stButton > button {
  background: var(--warm-white) !important;
  color: var(--ink) !important;
  border: 1.5px solid var(--sand) !important;
  border-radius: var(--radius-sm) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.82rem !important;
  font-weight: 500 !important;
  padding: 7px 14px !important;
  transition: var(--tr) !important;
  box-shadow: var(--shadow-sm) !important;
}

.stButton > button:hover {
  background: var(--accent-light) !important;
  border-color: var(--accent) !important;
  color: var(--accent) !important;
  box-shadow: var(--shadow-md) !important;
  transform: translateY(-1px) !important;
}

.stButton > button:active {
  transform: translateY(0) !important;
}

/* Primary / clear-chat button */
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
  background: var(--ink) !important;
  color: var(--cream) !important;
  border-color: var(--ink) !important;
}

[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
  background: var(--accent) !important;
  border-color: var(--accent) !important;
  color: #fff !important;
}

/* ── Toggle ─────────────────────────────────────────── */
.stToggle > label {
  font-size: 0.82rem !important;
  font-weight: 500 !important;
  color: var(--ink) !important;
}

[data-testid="stToggle"] span[data-checked="true"] {
  background-color: var(--accent) !important;
}

/* ── Slider ─────────────────────────────────────────── */
.stSlider > label {
  font-size: 0.82rem !important;
  font-weight: 500 !important;
  color: var(--ink) !important;
}

.stSlider [data-baseweb="slider"] div[role="slider"] {
  background: var(--accent) !important;
  border-color: var(--accent) !important;
}

.stSlider [data-baseweb="slider"] div[data-testid="stSliderTrack"] {
  background: var(--sand) !important;
}

/* Slider value label */
.stSlider [data-testid="stThumbValue"] {
  background: var(--accent-light) !important;
  color: var(--accent) !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.72rem !important;
  border-radius: 4px !important;
  padding: 2px 6px !important;
}

/* ── File uploader ──────────────────────────────────── */
[data-testid="stFileUploader"] {
  background: var(--cream) !important;
  border: 1.5px dashed var(--tan) !important;
  border-radius: var(--radius) !important;
  padding: 12px !important;
  transition: var(--tr) !important;
}

[data-testid="stFileUploader"]:hover {
  border-color: var(--accent) !important;
  background: var(--accent-light) !important;
}

[data-testid="stFileUploader"] label {
  font-size: 0.8rem !important;
  color: var(--ink-60) !important;
}

[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] {
  font-size: 0.78rem !important;
}

/* ── Chat messages ──────────────────────────────────── */
.stChatMessage {
  background: var(--warm-white) !important;
  border: 1px solid var(--sand) !important;
  border-radius: var(--radius) !important;
  box-shadow: var(--shadow-sm) !important;
  padding: 14px 18px !important;
  margin-bottom: 10px !important;
  animation: slide-up 0.28s ease !important;
}

@keyframes slide-up {
  from { opacity: 0; transform: translateY(6px); }
  to   { opacity: 1; transform: translateY(0);   }
}

/* User bubble — darker */
.stChatMessage[data-testid="stChatMessageUser"] {
  background: var(--ink) !important;
  border-color: var(--ink) !important;
  color: var(--cream) !important;
}

.stChatMessage[data-testid="stChatMessageUser"] p,
.stChatMessage[data-testid="stChatMessageUser"] .stMarkdown {
  color: var(--cream) !important;
}

/* Avatar icons */
.stChatMessage [data-testid="chatAvatarIcon-assistant"] {
  background: var(--accent) !important;
  color: #fff !important;
  border-radius: 8px !important;
}

.stChatMessage [data-testid="chatAvatarIcon-user"] {
  background: var(--bark) !important;
  color: var(--cream) !important;
  border-radius: 8px !important;
}

/* Message text */
.stChatMessage p {
  font-size: 0.88rem !important;
  line-height: 1.65 !important;
  color: var(--ink) !important;
}

/* Inline code in messages */
.stChatMessage code {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.8rem !important;
  background: var(--cream) !important;
  color: var(--accent) !important;
  border-radius: 4px !important;
  padding: 1px 5px !important;
}

/* ── Chat input ─────────────────────────────────────── */
[data-testid="stChatInput"] {
  border: 1.5px solid var(--sand) !important;
  border-radius: 12px !important;
  background: var(--warm-white) !important;
  box-shadow: var(--shadow-sm) !important;
  transition: var(--tr) !important;
}

[data-testid="stChatInput"]:focus-within {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(196,83,19,.08), var(--shadow-sm) !important;
}

[data-testid="stChatInput"] textarea {
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.88rem !important;
  color: var(--ink) !important;
  background: transparent !important;
}

[data-testid="stChatInput"] textarea::placeholder {
  color: var(--ink-30) !important;
}

/* Send button inside chat input */
[data-testid="stChatInput"] button {
  background: var(--ink) !important;
  border-radius: 8px !important;
  color: var(--cream) !important;
  transition: var(--tr) !important;
}

[data-testid="stChatInput"] button:hover {
  background: var(--accent) !important;
}

/* ── Main area top padding ───────────────────────────── */
.main .block-container {
  padding-top: 1.6rem !important;
  max-width: 860px !important;
}

/* ── Page title ─────────────────────────────────────── */
h1 {
  font-family: 'Libre Baskerville', serif !important;
  font-size: 1.5rem !important;
  font-weight: 700 !important;
  color: var(--ink) !important;
  letter-spacing: -0.025em !important;
  margin-bottom: 0.15rem !important;
}

/* Subtitle caption below title */
h1 + p, .stCaption {
  font-size: 0.8rem !important;
  color: var(--ink-30) !important;
  letter-spacing: 0.02em !important;
}

/* ── Metric / debug cards ───────────────────────────── */
[data-testid="stMetric"] {
  background: var(--warm-white) !important;
  border: 1px solid var(--sand) !important;
  border-radius: var(--radius) !important;
  padding: 10px 14px !important;
  box-shadow: var(--shadow-sm) !important;
}

[data-testid="stMetric"] label {
  font-size: 0.68rem !important;
  text-transform: uppercase !important;
  letter-spacing: 0.08em !important;
  color: var(--ink-30) !important;
  font-weight: 600 !important;
}

[data-testid="stMetric"] [data-testid="stMetricValue"] {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 1.1rem !important;
  color: var(--ink) !important;
}

/* ── Expander (retrieval debug) ─────────────────────── */
.streamlit-expanderHeader {
  background: var(--warm-white) !important;
  border: 1px solid var(--sand) !important;
  border-radius: var(--radius) !important;
  font-size: 0.82rem !important;
  font-weight: 600 !important;
  color: var(--ink) !important;
  padding: 10px 14px !important;
}

.streamlit-expanderContent {
  background: var(--cream) !important;
  border: 1px solid var(--sand) !important;
  border-top: none !important;
  border-radius: 0 0 var(--radius) var(--radius) !important;
  padding: 12px 14px !important;
}

/* ── Success / error alerts ─────────────────────────── */
.stAlert[data-baseweb="notification"][kind="positive"],
div[data-testid="stNotification"][data-baseweb-kind="positive"] {
  background: var(--green-light) !important;
  border-left: 4px solid var(--green) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--green) !important;
  font-size: 0.8rem !important;
}

div[data-testid="stNotification"][data-baseweb-kind="negative"] {
  background: #fdecea !important;
  border-left: 4px solid #c0392b !important;
  border-radius: var(--radius-sm) !important;
  font-size: 0.8rem !important;
}

/* ── Subheader / write text in sidebar ──────────────── */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] b {
  color: var(--ink) !important;
  font-weight: 600 !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li {
  font-size: 0.78rem !important;
  color: var(--ink-60) !important;
  font-family: 'JetBrains Mono', monospace !important;
  background: var(--cream);
  border: 1px solid var(--sand);
  border-radius: var(--radius-sm);
  padding: 5px 9px;
  margin-bottom: 4px;
  list-style: none;
}

/* ── Scrollbars ─────────────────────────────────────── */
::-webkit-scrollbar       { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--tan); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--bark); }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# PAGE HEADER
# ==========================================================

st.title("Enterprise RAG Platform")
st.caption("Hybrid semantic + keyword retrieval · Streaming · Live eval scores")

# ==========================================================
# SIDEBAR — SYSTEM CONTROLS
# ==========================================================

st.sidebar.header("⚙️ System Controls")

use_streaming = st.sidebar.toggle("Enable Streaming", value=True)
top_k = st.sidebar.slider("Top-K Retrieval", 1, 10, 5)
clear_chat = st.sidebar.button("🗑 Clear Chat", use_container_width=True)

st.sidebar.divider()

# ==========================================================
# SIDEBAR — DOCUMENT MANAGEMENT
# ==========================================================

st.sidebar.header("📄 Document Management")

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF or TXT",
    type=["pdf", "txt"]
)

if uploaded_file:
    files = {"file": uploaded_file}
    response = requests.post(UPLOAD_URL, files=files)
    if response.status_code == 200:
        st.sidebar.success("Uploaded successfully!")
    else:
        st.sidebar.error("Upload failed.")

if st.sidebar.button("🔄 Rebuild Index", use_container_width=True):
    r = requests.post(REBUILD_URL)
    if r.status_code == 200:
        st.sidebar.success("Index rebuilt successfully!")
    else:
        st.sidebar.error("Index rebuild failed.")

st.sidebar.subheader("📁 Uploaded Documents")

try:
    docs = requests.get(DOCS_URL).json()
    for doc in docs:
        col1, col2 = st.sidebar.columns([3, 1])
        col1.write(doc)
        if col2.button("❌", key=doc):
            requests.delete(f"{DOCS_URL}/{doc}")
            st.rerun()
except:
    st.sidebar.write("No documents found.")

st.sidebar.divider()

# ==========================================================
# SESSION STATE
# ==========================================================

if clear_chat:
    st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "metadata" not in st.session_state:
    st.session_state.metadata = {}

# ==========================================================
# CHAT DISPLAY
# ==========================================================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==========================================================
# CHAT INPUT
# ==========================================================

query = st.chat_input("Ask a question about enterprise documents…")

if query:

    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):

        response_placeholder = st.empty()
        full_response = ""

        try:

            if use_streaming:
                with requests.post(
                    STREAM_URL,
                    json={"query": query, "top_k": top_k},
                    stream=True,
                    timeout=300
                ) as r:

                    for line in r.iter_lines():
                        if line:
                            decoded = line.decode("utf-8")

                            if not decoded.startswith("data:"):
                                continue

                            data = decoded.replace("data: ", "")

                            if data == "[DONE]":
                                break

                            try:
                                parsed = json.loads(data)
                                token = parsed.get("token", "")
                            except:
                                token = data

                            full_response += token
                            response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)

            else:
                r = requests.post(API_URL, json={"query": query, "top_k": top_k})
                data = r.json()
                full_response = data["answer"]
                st.session_state.metadata = data
                response_placeholder.markdown(full_response)

        except Exception as e:
            full_response = "⚠️ Backend error."
            st.error(str(e))

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )

    # Fetch metadata after streaming
    if use_streaming:
        try:
            meta = requests.post(API_URL, json={"query": query}).json()
            st.session_state.metadata = meta
        except:
            pass

# ==========================================================
# RETRIEVAL DEBUG PANEL
# ==========================================================

if st.session_state.metadata:

    meta = st.session_state.metadata

    with st.expander("📊 Retrieval Debug", expanded=True):

        # Top metrics row
        col1, col2, col3 = st.columns(3)

        if "latency_seconds" in meta:
            col1.metric("⏱ Latency", f"{meta['latency_seconds']}s")

        if "faithfulness" in meta:
            col2.metric("✅ Faithfulness", f"{meta['faithfulness']:.0%}")

        if "cached" in meta:
            col3.metric("⚡ Cached", "Yes" if meta["cached"] else "No")

        # Sources list
        if "sources" in meta:
            st.markdown("**Sources**")
            src_items = "\n".join(f"- `{src}`" for src in meta["sources"])
            st.markdown(src_items)