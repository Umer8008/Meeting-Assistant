"""
RAG (Retrieval-Augmented Generation) engine for meeting Q&A.

Allows users to ask natural-language questions about a meeting transcript.
Uses the vector store retriever to find relevant transcript chunks,
then feeds them as context to the LLM for grounded answer generation.
"""

from core.summarize import get_llm
from core.vector_store import get_retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def _format_docs(docs):
    """Joins retrieved document chunks into a single context string."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def ask_question(question: str, vector_store) -> dict:
    """Answers a question using RAG over the meeting transcript.

    Args:
        question: The user's natural-language question.
        vector_store: A Chroma vector store built from the transcript.

    Returns:
        dict with keys:
          - "answer": The LLM-generated answer grounded in the transcript.
          - "sources": List of relevant transcript excerpts used as context.
    """
    retriever = get_retriever(vector_store, k=4)

    # Retrieve relevant chunks first (for returning as sources)
    source_docs = retriever.invoke(question)
    context = _format_docs(source_docs)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful meeting assistant. Answer the user's question "
            "based ONLY on the meeting transcript context provided below. "
            "If the answer cannot be found in the context, say so clearly. "
            "Do not make up information.\n\n"
            "Meeting Transcript Context:\n{context}"
        ),
        ("human", "{question}"),
    ])

    llm = get_llm()
    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "question": question
    })

    return {
        "answer": answer,
        "sources": [doc.page_content for doc in source_docs]
    }
