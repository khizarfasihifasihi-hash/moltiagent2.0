import os
import io
import re
import streamlit as st
import streamlit.components.v1 as components
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="Avatar AI Experts", page_icon="💧", layout="centered")

# ---------------------------------------------------------------------------
# Design system: "Boarding Pass"
# Picking a persona = boarding a flight into that expert's world. The header
# renders as a literal ticket stub with a channel number, a torn-perforation
# divider, and a rotated stamp. Everything else stays quiet around it.
# ---------------------------------------------------------------------------
DESIGN_CSS = """
<style>
@import url('https://googleapis.com');

/* Theme source of truth: Streamlit's OWN live CSS variables */
:root {
    --ink: #1B1230;
    --marigold: #2E86FF;
    --coral: #FF5F87;

    --bg-app: var(--background-color, #1B1230);
    --panel: var(--secondary-background-color, #241A3D);
    --text-primary: var(--text-color, #F5F1FA);

    --panel-2: color-mix(in srgb, var(--panel) 85%, var(--marigold) 15%);
    --text-muted: color-mix(in srgb, var(--text-primary) 65%, var(--bg-app) 35%);
    --border: color-mix(in srgb, var(--text-primary) 18%, var(--bg-app) 82%);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--text-primary);
}

.stApp {
    background: var(--bg-app);
}

/* Smooth, quiet transitions */
.stApp, section[data-testid="stSidebar"], .ticket, div[data-testid="stChatMessage"],
[data-testid="stFileUploaderDropzone"], [data-testid="stChatInput"] {
    transition: background-color 0.25s ease, border-color 0.25s ease, color 0.25s ease, box-shadow 0.25s ease;
}
@media (prefers-reduced-motion: reduce) {
    .stApp, section[data-testid="stSidebar"], .ticket, div[data-testid="stChatMessage"],
    [data-testid="stFileUploaderDropzone"], [data-testid="stChatInput"] {
        transition: none !important;
        animation: none !important;
    }
}

/* Sidebar = departure board */
section[data-testid="stSidebar"] {
    background-color: var(--panel);
    border-right: 1px dashed var(--border);
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    font-family: 'Fraunces', serif;
    font-weight: 700;
    letter-spacing: 0.01em;
    color: var(--text-primary) !important;
}
section[data-testid="stSidebar"] label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted) !important;
}

/* Boarding pass header */
.ticket {
    display: flex;
    margin-bottom: 1.6rem;
    border-radius: 14px;
    overflow: hidden;
    background: var(--panel);
    position: relative;
    border: 1px solid var(--border);
}
.ticket-stub {
    flex: 0 0 92px;
    background: var(--marigold);
    color: var(--ink);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    font-family: 'IBM Plex Mono', monospace;
    padding: 0.8rem 0.4rem;
    position: relative;
}
.ticket-stub .no-label {
    font-size: 0.62rem;
    letter-spacing: 0.1em;
    opacity: 0.75;
}
.ticket-stub .no-value {
    font-size: 1.6rem;
    font-weight: 600;
    line-height: 1.1;
}
.ticket-perf {
    flex: 0 0 0;
    border-left: 2px dashed var(--ink);
    position: relative;
}
.ticket-main {
    flex: 1;
    padding: 1.1rem 1.4rem;
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
    min-width: 0;
}
.ticket-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
}
.ticket-title {
    font-family: 'Fraunces', serif;
    font-weight: 700;
    font-size: 1.85rem;
    line-height: 1.12;
    margin: 0;
    color: var(--text-primary);
}
.ticket-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 0.92rem;
    color: var(--text-muted);
}
.ticket-stamp {
    position: absolute;
    top: 14px;
    right: 18px;
    transform: rotate(-9deg);
    border: 2px solid var(--coral);
    color: var(--coral);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    font-weight: 600;
    padding: 3px 9px;
    border-radius: 6px;
    text-transform: uppercase;
}

/* Top navbar */
.topnav {
    width: 100%;
    background: var(--panel);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.7rem 1.4rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    border: 1px solid var(--border);
}
.topnav-brand {
    font-family: 'Fraunces', serif;
    font-weight: 700;
    font-size: 1.15rem;
    color: var(--text-primary);
}
.topnav-status {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    color: var(--coral);
    background: rgba(255, 95, 135, 0.1);
    padding: 4px 10px;
    border-radius: 999px;
}
</style>
"""
st.markdown(DESIGN_CSS, unsafe_allow_stdio=False, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Core Components Initialization (With Built-in 429 Retry Handlers)
# ---------------------------------------------------------------------------

# Fetch api key cleanly from environment or streamlit secrets
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY") or st.secrets.get("MISTRAL_API_KEY", "")

if not MISTRAL_API_KEY:
    st.warning("⚠️ Mistral API Key not found. Please set your MISTRAL_API_KEY environment variable.")

# 1. Initialize LLM with automatic 429 Retry handling
llm = ChatMistralAI(
    model="mistral-large-latest",
    api_key=MISTRAL_API_KEY,
    temperature=0.2,
    max_retries=5  # 👈 FIX: Automatically handles 429 rate limits via LangChain's tenacity wrapper
)

# 2. Initialize Embeddings with automatic 429 Retry handling 
embeddings_model = MistralAIEmbeddings(
    model="mistral-embed",
    api_key=MISTRAL_API_KEY,
    max_retries=5  # 👈 FIX: Prevents application crashing during large text processing pipelines
)

# 3. Text Splitter optimization
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # 👈 FIX: Larger chunk size means fewer batches are sent to the embedding endpoint
    chunk_overlap=150
)

# ---------------------------------------------------------------------------
# UI Render Layout
# ---------------------------------------------------------------------------

# Render Navbar
st.markdown("""
<div class="topnav">
    <div class="topnav-brand">💧 Avatar AI System</div>
    <div class="topnav-status">Ready for Departure</div>
</div>
""", unsafe_allow_html=True)

# Render Boarding Ticket Header Header
st.markdown("""
<div class="ticket">
    <div class="ticket-stub">
        <div class="no-label">GATE</div>
        <div class="no-value">A4</div>
    </div>
    <div class="ticket-perf"></div>
    <div class="ticket-main">
        <div class="ticket-eyebrow">Expert Flight Manifest</div>
        <h1 class="ticket-title">Avatar AI Experts</h1>
        <p class="ticket-subtitle">Select an expert avatar below to board their intelligence space.</p>
        <div class="ticket-stamp">Validated</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Manifest setup
with st.sidebar:
    st.header("🛄 Passenger Manifest")
    uploaded_file = st.file_uploader("Drop a context brief (.txt format)", type=["txt"])
    
    st.header("⚙️ System Control")
    expert_persona = st.selectbox(
        "Select your Destination Expert",
        ["Technical Architect", "Design System Lead", "Product Strategist"]
    )

# App Content & Processing Engine
if uploaded_file is not None:
    stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
    raw_text = stringio.read()
    
    # Process steps wrapper
    with st.spinner("Processing flight documents..."):
        try:
            chunks = text_splitter.split_text(raw_text)
            # Safe FAISS instantiation protected by the underlying max_retries parameter
            vector_store = FAISS.from_texts(chunks, embeddings_model)
            st.sidebar.success(f"✓ Context processed cleanly into {len(chunks)} fragments.")
        except Exception as e:
            st.error(f"Failed to generate context pipeline: {e}")

# Main Chat interaction placeholder setup
if "messages" not in st.session_state:
    st.session_state.messages = [AIMessage(content=f"Welcome aboard! I am your {expert_persona}. How can I assist you with your project route today?")]

for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)

if prompt := st.chat_input("Enter your request here..."):
    st.chat_message("user").write(prompt)
    st.session_state.messages.append(HumanMessage(content=prompt))
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Built-in max_retries will step in automatically if Mistral issues a 429 error code here
                response = llm.invoke(st.session_state.messages)
                st.write(response.content)
                st.session_state.messages.append(AIMessage(content=response.content))
            except Exception as e:
                st.error(f"An unexpected connection issue occurred: {e}")
