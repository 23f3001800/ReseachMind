"""Vector store module — FAISS-based semantic search over document chunks.

Uses HuggingFace sentence-transformers for free, local embeddings.
Falls back to a simple keyword search if dependencies are missing.
"""

from typing import List, Dict
from core.logger import get_logger

logger = get_logger(__name__)

# Global vector store instance (lazy-initialized)
_vector_store = None
_embeddings = None


def _get_embeddings():
    """Get or create the embedding model (cached globally)."""
    global _embeddings
    if _embeddings is None:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            _embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
            )
            logger.info("Loaded HuggingFace embeddings: all-MiniLM-L6-v2")
        except ImportError:
            logger.warning(
                "langchain-huggingface not installed. "
                "Install with: pip install langchain-huggingface sentence-transformers"
            )
            raise
    return _embeddings


def add_documents(chunks: List[Dict], source: str = "upload"):
    """Add chunked documents to the FAISS vector store.

    Args:
        chunks: List of {"content": str, "index": int, ...} from rag.chunk_text()
        source: Identifier for the document source (e.g., filename)
    """
    global _vector_store

    try:
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document
    except ImportError:
        logger.error("faiss-cpu or langchain-community not installed")
        raise ImportError("Install: pip install faiss-cpu langchain-community")

    embeddings = _get_embeddings()

    # Convert chunks to LangChain Documents with metadata
    documents = [
        Document(
            page_content=chunk["content"],
            metadata={
                "source": source,
                "chunk_index": chunk["index"],
                "char_start": chunk.get("char_start", 0),
            },
        )
        for chunk in chunks
    ]

    if _vector_store is None:
        _vector_store = FAISS.from_documents(documents, embeddings)
        logger.info(f"Created FAISS index | docs={len(documents)} source={source}")
    else:
        _vector_store.add_documents(documents)
        logger.info(f"Added to FAISS index | docs={len(documents)} source={source}")


def search(query: str, k: int = 5) -> List[Dict]:
    """Search the vector store for the most relevant chunks.

    Args:
        query: The search query string.
        k: Number of top results to return.

    Returns:
        List of {"content": str, "source": str, "score": float, "chunk_index": int}
    """
    if _vector_store is None:
        logger.warning("No documents in vector store — returning empty results")
        return []

    results = _vector_store.similarity_search_with_score(query, k=k)

    output = []
    for doc, score in results:
        output.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "chunk_index": doc.metadata.get("chunk_index", -1),
            "score": round(float(score), 4),
        })

    logger.info(f"Vector search | query='{query[:60]}' results={len(output)} top_score={output[0]['score'] if output else 'N/A'}")
    return output


def get_context_for_query(query: str, k: int = 3) -> str:
    """Convenience: retrieve top-k chunks and format as context string for the LLM."""
    results = search(query, k=k)
    if not results:
        return ""

    context_parts = []
    for r in results:
        context_parts.append(
            f"[Source: {r['source']}, Chunk {r['chunk_index']}]\n{r['content']}"
        )

    return "\n\n---\n\n".join(context_parts)


def has_documents() -> bool:
    """Check if any documents have been indexed."""
    return _vector_store is not None


def clear_store():
    """Clear the vector store."""
    global _vector_store
    _vector_store = None
    logger.info("Vector store cleared")
