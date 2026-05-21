# You might need the following imports. Feel free to change it if you opt for different libraries.
from __future__ import annotations
import os
import glob as globmod
from typing import Any
import numpy as np
import faiss
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder
from anthropic import Anthropic

# Default configs
DEFAULT_DATA_DIR = "data"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_LLM_MODEL = "claude-haiku-4-5"
DEFAULT_CHUNK_SIZE = 256
DEFAULT_CHUNK_OVERLAP = 32
DEFAULT_TOP_K = 4
DEFAULT_OVERFETCH = 20

TAG_DOC_TYPE = {
    "/notes": "notes",
    "/sms": "sms",
    "/calendar": "calendar",
    "/email": "emails",
    "/emails": "emails",
}


def extract_filter(query: str) -> tuple[str, str | None]:
    """Detects a tag in the query and returns clean_query, doc_type or None."""
    words = query.split() # split gets us every word
    
    # initialize returns
    doc_type = None
    kept = []

    # go through all words
    for word in words:
        tag = word.lower()

        # and check if they match a tag
        if doc_type is None and tag in TAG_DOC_TYPE:
            doc_type = TAG_DOC_TYPE[tag]
        else:
            kept.append(word)

    # return the found doc type and the cleaned query without the tag words
    return " ".join(kept), doc_type


def rerank(
        query: str,
        results: list[dict],
        crossencoder: CrossEncoder,
        k: int,
) -> list[dict]:
    """Re-ranks results with a CrossEncoder and returns top k."""

    # no results
    if not results:
        return results
    
    # pair the query with each result text
    pairs = [(query, r["text"]) for r in results]

    # get scores for each pair
    scores = crossencoder.predict(pairs)
    
    # attach scores to results
    for r, s in zip(results, scores):
        r["score"] = float(s)
    
    # sort by score and return the top k
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:k]


def _parse_int_setting(name: str, value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer; got {value!r}") from exc
    return parsed


def resolve_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolves runtime configuration with defaults and typed settings."""
    config = config or {}

    resolved = {
        "api_key": config.get("api_key", None),
        "base_url": config.get("base_url", None),
        "model": config.get("model") or DEFAULT_LLM_MODEL,
        "embedding_model": config.get("embedding_model") or DEFAULT_EMBEDDING_MODEL,
        "top_k": _parse_int_setting(
            "TOP_K",
            config.get("top_k") or DEFAULT_TOP_K,
        ),
        "chunk_size": _parse_int_setting(
            "CHUNK_SIZE",
            config.get("chunk_size") or DEFAULT_CHUNK_SIZE,
        ),
        "chunk_overlap": _parse_int_setting(
            "CHUNK_OVERLAP",
            config.get("chunk_overlap") or DEFAULT_CHUNK_OVERLAP,
        ),
    }

    if resolved["top_k"] <= 0:
        raise ValueError("TOP_K must be > 0")
    if resolved["chunk_size"] <= 0:
        raise ValueError("CHUNK_SIZE must be > 0")
    if resolved["chunk_overlap"] < 0:
        raise ValueError("CHUNK_OVERLAP must be >= 0")
    if resolved["chunk_overlap"] >= resolved["chunk_size"]:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    return resolved


def load_documents(data_dir: str = DEFAULT_DATA_DIR) -> list[Document]:
    """Loads documents from the personal data folders.

    The collection contains one LangChain Document per `.txt` file in the
    emails, notes, SMS, and calendar folders. Each document stores the file text
    as `page_content` and includes metadata for the source file path and
    document type.
    """
    docs = []
    for doc_type in ["emails", "notes", "sms", "calendar"]:
        pattern = os.path.join(data_dir, doc_type, "*.txt")
        for path in globmod.glob(pattern):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            metadata = {"source": path, "doc_type": doc_type}
            docs.append(Document(page_content=text, metadata=metadata))
    return docs


def split_documents(
        docs: list[Document],
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Splits documents into overlapping chunks.

    The resulting chunked Document objects use the configured chunk size and
    overlap while preserving the original document metadata.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return text_splitter.split_documents(docs)


def build_index(
        chunks: list[Document],
        embedding_model: SentenceTransformer,
) -> faiss.IndexFlatIP:
    """Creates a FAISS inner-product index for embedded document chunks.

    The index contains normalized float32 embeddings generated from each
    chunk's text with the provided embedding model.
    """
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embedding_model.encode(texts, normalize_embeddings=True).astype(np.float32)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def retrieve(
        query: str,
        index: faiss.IndexFlatIP,
        model: SentenceTransformer,
        chunks: list[Document],
        k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """Gets the most relevant chunks for a query.

    Results are ordered by similarity and include the chunk text, similarity
    score, and metadata for each matching chunk.
    """
    query_vec = model.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(query_vec, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append({
            "text": chunks[idx].page_content,
            "score": float(score),
            "metadata": chunks[idx].metadata,
        })
    return results


SYSTEM_PROMPT = """You are a personal digital assistant. Answer the user's question \
using ONLY the provided context from their personal documents \
(emails, notes, SMS, and calendar events). Follow these rules:
- If the context doesn't contain the answer, say "I don't have enough \
information to answer this question."
- Be concise and precise.
- Do not use prior knowledge outside of the context.
- When referencing information, mention the source type and file."""


class Assistant:
    """Stateful RAG assistant.

    The assistant owns the pipeline components, resolved configuration, and
    conversation history. Questions are answered with retrieved document context
    and the configured chat model.
    """

    def __init__(
            self,
            index: faiss.IndexFlatIP,
            model: SentenceTransformer,
            chunks: list[Document],
            client: Anthropic,
            config: dict[str, Any] | None = None,
    ) -> None:
        self.index = index
        self.model = model
        self.chunks = chunks
        self.client = client
        self.config = resolve_config(config)
        self.llm_model = self.config["model"]
        self.top_k = self.config["top_k"]
        self.cross_encoder = CrossEncoder(DEFAULT_RERANK_MODEL)
        self.history: list[dict[str, str]] = []

    def ask(self, question: str, k: int | None = None) -> str:
        """Generates an answer from the retrieved context and conversation history.

        The current question is combined with relevant document chunks, previous
        conversation messages, and the system prompt. The assistant response is
        appended to history alongside the user message.
        """
        num_results = k if k is not None else self.top_k

        # overfetching
        # first we exxtract the question and the doc type
        clean_question, doc_type = extract_filter(question)

        # then, we get more results (num_results * 5), clamped to DEFAULT_OVERFETCH, to have more candidates for the re-ranker and filtering
        fetch_k = min(max(num_results * 5, num_results), DEFAULT_OVERFETCH)
        search_results = retrieve(clean_question, self.index, self.model, self.chunks, fetch_k)

        # if a doc type was specified, filter results by it 
        if doc_type is not None:
            search_results = [r for r in search_results if r["metadata"].get("doc_type") == doc_type]

        # then rerank
        search_results = rerank(clean_question, search_results, self.cross_encoder, num_results)

        context_parts = []
        source_files = []
        for result in search_results:
            context_parts.append(result["text"])
            src = result["metadata"].get("source", "unknown")
            if src not in source_files:
                source_files.append(src)

        context_block = "\n\n---\n\n".join(context_parts)

        user_content = f"Context:\n{context_block}\n\nQuestion: {question}"

        self.history.append({"role": "user", "content": user_content})

        response = self.client.messages.create(
            model=self.llm_model,
            system=SYSTEM_PROMPT,
            messages=self.history,
            max_tokens=1024,
        )

        answer = response.content[0].text
        self.history.append({"role": "assistant", "content": answer})

        ref_list = "\n".join(f"- {src}" for src in source_files)
        return f"{answer}\n\nReference:\n{ref_list}"

    def clear_history(self) -> None:
        """Empties the conversation history."""
        self.history.clear()

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> Assistant:
        """Initializes the components required by the assistant and instantiates it

        The pipeline includes resolved configuration, loaded documents, chunked
        documents, an embedding model, a FAISS index, and an OpenAI-compatible
        client.
        """
        resolved_config = resolve_config(config)

        print("Loading documents...")
        docs = load_documents()
        print(f"  Loaded {len(docs)} documents")

        print("Splitting into chunks...")
        chunks = split_documents(
            docs,
            chunk_size=resolved_config["chunk_size"],
            chunk_overlap=resolved_config["chunk_overlap"],
        )
        print(f"  Created {len(chunks)} chunks")

        embedding_model = SentenceTransformer(resolved_config["embedding_model"])

        print("Building FAISS index...")
        index = build_index(chunks, embedding_model)
        print(f"  Indexed {index.ntotal} vectors (dim={index.d})")

        client_kwargs = {}
        if resolved_config["api_key"]:
            client_kwargs["api_key"] = resolved_config["api_key"]
        if resolved_config["base_url"]:
            client_kwargs["base_url"] = resolved_config["base_url"]
        client = Anthropic(**client_kwargs)

        print("Ready!\n")
        return cls(index, embedding_model, chunks, client, resolved_config)
