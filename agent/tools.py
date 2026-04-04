# agent/tools.py
import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


from dotenv import load_dotenv
load_dotenv()

os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# Web search tool
search_tool = TavilySearchResults(
    max_results=4,
    include_answer=True,
    include_raw_content=True,
    max_characters=3000           # limit content per result to save tokens
)

# RAG tool (optional — activated if user uploads a doc)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def build_rag_retriever(documents):
    """Build a retriever from uploaded documents."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10}
    )

def search_web(query: str) -> list:
    """Run a single web search, return structured results."""
    try:
        results = search_tool.invoke(query)
        return results
    except Exception as e:
        return [{"content": f"Search failed: {str(e)}", "url": ""}]

def search_rag(retriever, query: str) -> str:
    """Query the RAG knowledge base."""
    if retriever is None:
        return ""
    try:
        docs = retriever.invoke(query)
        return "\n\n".join([d.page_content for d in docs])
    except Exception:
        return ""