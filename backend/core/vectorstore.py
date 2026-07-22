"""Vector store — FAISS semantic search over document chunks.

Embeddings come from fastembed (ONNX runtime), deliberately not
sentence-transformers: the latter pulls in torch and takes the image from
roughly 200 MB to 2.5 GB, which is the difference between an image that cold
starts in seconds and one that doesn't fit a small container at all. Same
model family, same quality tier, ~50 MB of runtime.

The index is persisted to disk so uploads survive a restart or scale-to-zero.
Point DATA_DIR at a mounted volume in production; on a container's own
filesystem it is still lost when the container is replaced.
"""

import json
import os
import threading
from typing import Dict, List

from core.logger import get_logger

logger = get_logger(__name__)

# Embedding model. 384 dimensions, strong quality-per-byte for short chunks.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

def _data_dir() -> str:
    """Storage root, read at call time so tests can point DATA_DIR elsewhere."""
    return os.getenv("DATA_DIR") or "data"


DATA_DIR = _data_dir()
INDEX_DIR = os.path.join(DATA_DIR, "vectorstore")
CATALOG_PATH = os.path.join(DATA_DIR, "documents.json")

_vector_store = None
_embeddings = None
_lock = threading.Lock()


def dependencies_available() -> bool:
    """True only if the heavy vector dependencies are genuinely importable.

    This module imports cleanly without them — they are imported lazily inside
    the functions that need them — so `import vectorstore` succeeding is NOT
    evidence that indexing will work. Callers gating a feature flag on RAG
    support must ask this, or they advertise a capability that fails at first use.
    """
    try:
        import faiss  # noqa: F401
        from fastembed import TextEmbedding  # noqa: F401
        from langchain_community.vectorstores import FAISS  # noqa: F401
        return True
    except ImportError:
        return False


def _embeddings_base():
    """The Embeddings ABC from langchain-core.

    Subclassing this is not optional: FAISS does an isinstance check and
    silently degrades to calling the object as a bare function if it fails,
    which blows up at query time rather than at construction.
    """
    from langchain_core.embeddings import Embeddings

    return Embeddings


class FastEmbedAdapter(_embeddings_base()):  # type: ignore[misc]
    """LangChain Embeddings implementation backed by fastembed.

    Written against the core ABC rather than pulling a community integration:
    this project has already been broken twice by integration packages changing
    their constructor contract silently, and Embeddings itself is a stable
    two-method interface.
    """

    def __init__(self, model_name: str = EMBED_MODEL):
        from fastembed import TextEmbedding

        # FASTEMBED_CACHE points at the model baked into the image. Without it
        # fastembed defaults to a temp dir and re-downloads on every cold start.
        cache_dir = os.getenv("FASTEMBED_CACHE") or None
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [vec.tolist() for vec in self._model.embed(list(texts))]

    def embed_query(self, text: str) -> List[float]:
        return list(self._model.embed([text]))[0].tolist()


def _get_embeddings() -> "FastEmbedAdapter":
    """Get or create the embedding model (cached — model load is expensive)."""
    global _embeddings
    if _embeddings is None:
        _embeddings = FastEmbedAdapter()
        logger.info(f"Loaded fastembed embeddings | model={EMBED_MODEL}")
    return _embeddings


# ── Persistence ───────────────────────────────────────────
def _save_index() -> None:
    """Write the FAISS index to disk so it survives a restart."""
    if _vector_store is None:
        return
    try:
        os.makedirs(INDEX_DIR, exist_ok=True)
        _vector_store.save_local(INDEX_DIR)
        logger.info(f"Vector index persisted | path={INDEX_DIR}")
    except Exception as e:
        # Never fail an upload because persistence failed — the in-memory index
        # is still usable for this process.
        logger.warning(f"Could not persist vector index | error={e}")


def _load_index() -> bool:
    """Restore a previously persisted index. True if one was loaded."""
    global _vector_store
    if not os.path.isdir(INDEX_DIR):
        return False
    try:
        from langchain_community.vectorstores import FAISS

        _vector_store = FAISS.load_local(
            INDEX_DIR,
            _get_embeddings(),
            # We wrote this file ourselves on our own volume; it is not user input.
            allow_dangerous_deserialization=True,
        )
        logger.info(f"Vector index restored from disk | path={INDEX_DIR}")
        return True
    except Exception as e:
        logger.warning(f"Could not restore vector index | error={e}")
        return False


def load_catalog() -> Dict[str, Dict]:
    """Document metadata, persisted alongside the index."""
    try:
        with open(CATALOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_catalog(catalog: Dict[str, Dict]) -> None:
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(CATALOG_PATH, "w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not persist document catalog | error={e}")


def initialize() -> None:
    """Restore any persisted index at startup. Safe to call more than once."""
    with _lock:
        if _vector_store is None:
            _load_index()


# ── Indexing and search ───────────────────────────────────
def add_documents(chunks: List[Dict], source: str = "upload"):
    """Add chunked documents to the FAISS vector store and persist the result.

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
        raise ImportError("Install: pip install -r requirements-rag.txt")

    embeddings = _get_embeddings()

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

    with _lock:
        if _vector_store is None:
            _load_index()

        if _vector_store is None:
            _vector_store = FAISS.from_documents(documents, embeddings)
            logger.info(f"Created FAISS index | docs={len(documents)} source={source}")
        else:
            _vector_store.add_documents(documents)
            logger.info(f"Added to FAISS index | docs={len(documents)} source={source}")

        _save_index()


def search(query: str, k: int = 5) -> List[Dict]:
    """Search the vector store for the most relevant chunks.

    Returns:
        List of {"content": str, "source": str, "score": float, "chunk_index": int}
    """
    if _vector_store is None:
        initialize()
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

    logger.info(
        f"Vector search | query='{query[:60]}' results={len(output)} "
        f"top_score={output[0]['score'] if output else 'N/A'}"
    )
    return output


def get_context_for_query(query: str, k: int = 3) -> str:
    """Convenience: retrieve top-k chunks and format as context string for the LLM."""
    results = search(query, k=k)
    if not results:
        return ""

    return "\n\n---\n\n".join(
        f"[Source: {r['source']}, Chunk {r['chunk_index']}]\n{r['content']}"
        for r in results
    )


def has_documents() -> bool:
    """Check if any documents have been indexed."""
    if _vector_store is None:
        initialize()
    return _vector_store is not None


def clear_store():
    """Clear the vector store, on disk as well as in memory."""
    global _vector_store
    with _lock:
        _vector_store = None
        try:
            import shutil

            shutil.rmtree(INDEX_DIR, ignore_errors=True)
            if os.path.exists(CATALOG_PATH):
                os.remove(CATALOG_PATH)
        except Exception as e:
            logger.warning(f"Could not clear persisted index | error={e}")
    logger.info("Vector store cleared")
