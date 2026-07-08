"""
rag/chain.py — LangChain RetrievalQA chain.

Builds a chain that:
  1. Takes a natural language question
  2. Retrieves the top-K most semantically relevant chunks from Qdrant
  3. Passes them as context to GPT-4o-mini with a strict grounding prompt
  4. Returns the answer + the source document metadata
"""
from functools import lru_cache

# LangChain 0.1.x+ uses different import paths
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_openai import ChatOpenAI

from ..config import settings
from .vectorstore import get_vectorstore

# ── System prompt ─────────────────────────────────────────────────────────────
_TEMPLATE = """You are an enterprise document intelligence assistant.
Your only source of knowledge is the context below — do NOT use any prior knowledge.
If the answer is not present in the context, respond with exactly:
"I don't have enough information in the provided documents to answer that question."

Context:
{context}

Question: {question}

Answer (be concise and cite the document where possible):"""

PROMPT = ChatPromptTemplate.from_template(_TEMPLATE)


@lru_cache(maxsize=1)
def build_rag_chain():
    """
    Build and cache the RAG chain.
    Cached so we don't rebuild the LLM + retriever on every request.
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,          # deterministic — no hallucination drift
        openai_api_key=settings.openai_api_key,
        request_timeout=60,     # timeout in seconds
        max_retries=3,          # retry on transient failures
    )

    retriever = get_vectorstore().as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.top_k_chunks},
    )

    # New LangChain version
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
    )
    return rag_chain