"""RAG module — document upload, chunking, and vector retrieval.

Supports PDF, TXT, and Markdown files. Uses RecursiveCharacterTextSplitter
for intelligent chunking and FAISS for local vector storage.
"""

import os
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.logger import get_logger

logger = get_logger(__name__)


def load_document(file_path: str) -> str:
    """Load a document file and return its text content.

    Supports: .txt, .md, .pdf
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext in (".txt", ".md"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        logger.info(f"Loaded text file | path={file_path} chars={len(text)}")
        return text

    elif ext == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            pages = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            full_text = "\n\n".join(pages)
            logger.info(f"Loaded PDF | path={file_path} pages={len(reader.pages)} chars={len(full_text)}")
            return full_text
        except ImportError:
            logger.error("pypdf not installed — cannot read PDF files")
            raise ImportError("Install pypdf: pip install pypdf")

    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: .txt, .md, .pdf")


def load_text(text: str, source: str = "pasted") -> str:
    """Pass-through for raw text input."""
    logger.info(f"Loaded raw text | source={source} chars={len(text)}")
    return text


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Dict]:
    """Split text into overlapping chunks with metadata.

    Returns a list of dicts: {"content": str, "index": int, "char_start": int}
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_text(text)

    result = []
    char_offset = 0
    for i, chunk in enumerate(chunks):
        result.append({
            "content": chunk,
            "index": i,
            "char_start": text.find(chunk, char_offset),
        })
        char_offset = max(char_offset, text.find(chunk, char_offset))

    logger.info(
        f"Chunked text | total_chars={len(text)} "
        f"chunks={len(result)} chunk_size={chunk_size} overlap={chunk_overlap}"
    )
    return result
