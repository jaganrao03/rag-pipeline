"""
query.py — Takes a natural language question, finds relevant chunks from Pinecone,
and uses Claude to generate a grounded answer.

Pipeline:
  Question → embed question → Pinecone similarity search → Claude answer
"""

from langchain_anthropic import ChatAnthropic
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from langchain.schema import HumanMessage, SystemMessage

from src.config import (
    ANTHROPIC_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
    CLAUDE_MODEL,
    TOP_K,
)


def get_vector_store() -> PineconeVectorStore:
    """
    Connects to the existing Pinecone index (already populated by ingest.py).
    Returns a LangChain VectorStore object we can search.
    """
    embeddings = PineconeEmbeddings(
        model=EMBEDDING_MODEL,
        pinecone_api_key=PINECONE_API_KEY,
    )
    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=PINECONE_API_KEY,
    )


def retrieve_context(question: str, top_k: int = TOP_K) -> list:
    """
    Embeds the question and searches Pinecone for the most similar document chunks.

    How it works:
    1. The question is converted to a vector (same embedding model as ingestion)
    2. Pinecone finds the stored chunks whose vectors are closest to the question vector
    3. "Closest" means most semantically similar — not just keyword matching

    Returns a list of LangChain Document objects (chunk text + metadata).
    """
    vector_store = get_vector_store()
    # similarity_search returns chunks ordered from most to least relevant
    docs = vector_store.similarity_search(question, k=top_k)
    return docs


def build_prompt(question: str, context_docs: list) -> tuple[str, str]:
    """
    Assembles the system prompt and user message to send to Claude.

    The system prompt instructs Claude to answer ONLY from the provided context.
    This is what makes it "retrieval-augmented" — Claude's answer is grounded
    in your document, not just general training knowledge.
    """
    # Join all retrieved chunks into one context block
    context_text = "\n\n---\n\n".join(
        f"[Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in context_docs
    )

    system_prompt = (
        "You are a helpful assistant. Answer the user's question based ONLY on the "
        "provided document context below. If the context does not contain enough "
        "information to answer the question, say: "
        "'I don't have enough information in the document to answer that.' "
        "Do not make up information or use knowledge outside the provided context.\n\n"
        f"CONTEXT:\n{context_text}"
    )

    user_message = question

    return system_prompt, user_message


def ask_question(question: str) -> dict:
    """
    Main query function. Retrieves relevant context and asks Claude to answer.

    Returns a dict with:
    - question: the original question
    - answer: Claude's grounded answer
    - sources: list of chunks used (so users can verify the answer)
    """
    print(f"Question: {question}")

    # Step 1: Find relevant chunks from Pinecone
    print(f"  Retrieving top {TOP_K} relevant chunks from Pinecone...")
    context_docs = retrieve_context(question)

    # Step 2: Build the prompt
    system_prompt, user_message = build_prompt(question, context_docs)

    # Step 3: Ask Claude
    print(f"  Sending to Claude ({CLAUDE_MODEL})...")
    llm = ChatAnthropic(
        model=CLAUDE_MODEL,
        api_key=ANTHROPIC_API_KEY,
        max_tokens=1024,
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]
    response = llm.invoke(messages)
    answer = response.content

    # Step 4: Format sources for the response
    sources = [
        {
            "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
            "page": doc.metadata.get("page", None),
            "source": doc.metadata.get("source", "unknown"),
        }
        for doc in context_docs
    ]

    print("  Done.")
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
    }
