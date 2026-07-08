"""
frontend/app.py — Streamlit chat UI for the Enterprise RAG system.

Features:
  • PDF upload → async ingestion with live progress polling
  • Chat interface with persistent message history
  • Per-message latency, cache source, and similarity badges
  • Sidebar stats: cache entries, collection info, threshold display
  • Cache flush button
  • API key authentication
"""
from __future__ import annotations

import os
import time

import httpx
import streamlit as st

# Get API configuration from environment
API_BASE = os.getenv("API_BASE", "http://localhost:8001")
API_KEY = os.getenv("API_KEY", "")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Enterprise RAG — Document Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* Global reset */
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* Dark background */
.stApp { background-color: #0a0a0f; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #111118 !important;
    border-right: 1px solid #2a2a35 !important;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: #111118 !important;
    border: 1px solid #2a2a35 !important;
    border-radius: 12px !important;
    padding: 12px !important;
    margin-bottom: 8px !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #18181f !important;
    border: 1px solid #2a2a35 !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
}

/* Badge pill utility */
.badge-cache { color: #00e5c8; font-family: 'IBM Plex Mono', monospace; font-size: 11px;
    background: rgba(0,229,200,0.1); border: 1px solid rgba(0,229,200,0.3);
    padding: 2px 8px; border-radius: 20px; }
.badge-llm { color: #a89eff; font-family: 'IBM Plex Mono', monospace; font-size: 11px;
    background: rgba(124,109,255,0.1); border: 1px solid rgba(124,109,255,0.3);
    padding: 2px 8px; border-radius: 20px; }
.badge-latency { color: #ffb347; font-family: 'IBM Plex Mono', monospace; font-size: 11px;
    background: rgba(255,179,71,0.1); border: 1px solid rgba(255,179,71,0.3);
    padding: 2px 8px; border-radius: 20px; }

/* Code snippets */
code { font-family: 'IBM Plex Mono', monospace !important; color: #a89eff !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #7c6dff, #00e5c8) !important;
    color: #000 !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _get_headers() -> dict:
    """Get request headers with API key."""
    headers = {}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    return headers


def api_health() -> dict:
    try:
        r = httpx.get(f"{API_BASE}/v1/health", timeout=4, headers=_get_headers())
        return r.json()
    except Exception:
        return {}


def api_ingest(file_bytes: bytes, filename: str) -> dict:
    r = httpx.post(
        f"{API_BASE}/v1/ingest",
        files={"file": (filename, file_bytes, "application/pdf")},
        timeout=30,
        headers=_get_headers(),
    )
    r.raise_for_status()
    return r.json()


def api_task_status(task_id: str) -> dict:
    r = httpx.get(
        f"{API_BASE}/v1/task/{task_id}",
        timeout=10,
        headers=_get_headers(),
    )
    r.raise_for_status()
    return r.json()


def api_query(question: str) -> dict:
    r = httpx.post(
        f"{API_BASE}/v1/query",
        json={"question": question},
        timeout=60,
        headers=_get_headers(),
    )
    r.raise_for_status()
    return r.json()


def api_flush_cache() -> dict:
    r = httpx.delete(f"{API_BASE}/v1/cache", timeout=10, headers=_get_headers())
    r.raise_for_status()
    return r.json()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Enterprise RAG")
    st.caption("Document Intelligence Platform")
    st.divider()

    # ── API Key Configuration ───────────────────────────────────────────────────
    if not API_KEY:
        st.warning("⚠️ Set API_KEY in environment to connect to backend")
    
    # ── Health / stats ────────────────────────────────────────────────────────
    health = api_health()
    if health:
        qdrant = health.get("qdrant", {})
        cache = health.get("cache", {})
        cfg = health.get("config", {})

        col1, col2 = st.columns(2)
        col1.metric("📦 Vectors", qdrant.get("vectors_count", "—"))
        col2.metric("💾 Cached Q's", cache.get("cached_entries", "—"))

        st.caption(
            f"Collection: `{cfg.get('qdrant_collection', '—')}`  "
            f"· Threshold: `{cfg.get('similarity_threshold', '—')}`  "
            f"· Top-K: `{cfg.get('top_k_chunks', '—')}`"
        )
    else:
        st.warning("⚠️ API unreachable — is the backend running?")

    st.divider()

    # ── Document upload ───────────────────────────────────────────────────────
    st.markdown("### 📄 Ingest Document")
    uploaded = st.file_uploader("Choose a PDF file", type=["pdf"], label_visibility="collapsed")

    if uploaded:
        if st.button("🚀 Ingest Document", use_container_width=True):
            with st.status("Ingesting document…", expanded=True) as status_box:
                try:
                    resp = api_ingest(uploaded.getvalue(), uploaded.name)
                    task_id = resp["task_id"]
                    st.write(f"Task queued: `{task_id[:12]}…`")

                    # Poll until done
                    state_labels = {
                        "PENDING": "⏳ Waiting in queue…",
                        "STARTED": "🔄 Worker started…",
                        "PARSING": "📖 Parsing PDF…",
                        "CHUNKING": "✂️  Chunking text…",
                        "EMBEDDING": "🔢 Generating embeddings…",
                        "CLEANUP": "🧹 Cleaning up…",
                        "SUCCESS": "✅ Done!",
                        "FAILURE": "❌ Failed",
                    }

                    while True:
                        task = api_task_status(task_id)
                        state = task["status"]
                        st.write(state_labels.get(state, f"State: {state}"))

                        if state == "SUCCESS":
                            info = task.get("info", {})
                            status_box.update(label="✅ Ingestion complete!", state="complete")
                            st.success(
                                f"Ingested **{info.get('chunks_ingested', '?')} chunks** "
                                f"from **{info.get('pages', '?')} pages**"
                            )
                            if info.get("message"):
                                st.info(info["message"])
                            break
                        elif state == "FAILURE":
                            status_box.update(label="❌ Ingestion failed", state="error")
                            st.error(str(task.get("info", "Unknown error")))
                            break

                        time.sleep(1.5)

                except Exception as exc:
                    st.error(f"Error: {exc}")

    st.divider()
  
    # ── Cache management ──────────────────────────────────────────────────────
    st.markdown("### 🗑️ Cache Management")
    if st.button("Flush Semantic Cache", use_container_width=True):
        try:
            result = api_flush_cache()
            st.success(f"Cleared {result['deleted_entries']} cache entries.")
        except Exception as exc:
            st.error(str(exc))
  
    # ── Clear stuck queue ─────────────────────────────────────────────────────
    st.markdown("### 🧹 Queue Management")
    if st.button("Clear stuck queue", use_container_width=True):
        st.session_state.documents = []
        st.rerun()


# ── Main chat area ────────────────────────────────────────────────────────────
st.markdown("## 💬 Ask your documents anything")
st.caption("Upload a PDF in the sidebar, then ask questions below.")

# Session state for message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            meta = msg["meta"]
            source = meta.get("source", "")
            latency = meta.get("latency_ms", "—")
            sim = meta.get("similarity")

            badge_class = "badge-cache" if source == "CACHE HIT" else "badge-llm"
            badge_label = "⚡ CACHE HIT" if source == "CACHE HIT" else "🤖 LLM GENERATED"
            parts = [
                f'<span class="{badge_class}">{badge_label}</span>',
                f'<span class="badge-latency">⏱ {latency} ms</span>',
            ]
            if sim is not None:
                parts.append(f'<span class="badge-cache">≈ {sim:.3f}</span>')

            st.markdown(" &nbsp; ".join(parts), unsafe_allow_html=True)

            if meta.get("sources"):
                with st.expander("📎 Source documents"):
                    for src in meta["sources"]:
                        st.caption(f"• `{src.get('source', 'unknown')}` — doc `{src.get('doc_id', '?')[:8]}…`")


# Chat input
if prompt := st.chat_input("Ask a question about your documents…"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Query the API
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = api_query(prompt)
                answer = result["answer"]
                meta = {
                    "source": result.get("source"),
                    "latency_ms": result.get("latency_ms"),
                    "similarity": result.get("similarity"),
                    "sources": result.get("sources"),
                    "cached_query": result.get("cached_query"),
                }

                st.markdown(answer)

                source = meta["source"]
                latency = meta["latency_ms"]
                sim = meta["similarity"]

                badge_class = "badge-cache" if source == "CACHE HIT" else "badge-llm"
                badge_label = "⚡ CACHE HIT" if source == "CACHE HIT" else "🤖 LLM GENERATED"
                parts = [
                    f'<span class="{badge_class}">{badge_label}</span>',
                    f'<span class="badge-latency">⏱ {latency} ms</span>',
                ]
                if sim is not None:
                    parts.append(f'<span class="badge-cache">≈ {sim:.3f}</span>')

                st.markdown(" &nbsp; ".join(parts), unsafe_allow_html=True)

                if meta.get("sources"):
                    with st.expander("📎 Source documents"):
                        for src in meta["sources"]:
                            st.caption(f"• `{src.get('source', 'unknown')}` — doc `{src.get('doc_id', '?')[:8]}…`")

                if meta.get("cached_query") and source == "CACHE HIT":
                    st.caption(f"Matched cached query: *\"{meta['cached_query']}\"*")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "meta": meta,
                })

            except httpx.HTTPStatusError as exc:
                err = f"API error {exc.response.status_code}: {exc.response.text}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
            except Exception as exc:
                err = f"Connection error: {exc}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})