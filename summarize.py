"""
Meeting summarization and title generation.

Uses Google Gemini via LangChain for LLM operations.
Implements a map-reduce pattern for long transcripts:
  1. Split transcript into manageable chunks
  2. Summarize each chunk individually (map)
  3. Combine chunk summaries into a final summary (reduce)
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


def get_llm():
    """Returns a configured Gemini LLM instance.

    Uses GEMINI_API_KEY from environment variables.
    Model can be customized via GEMINI_MODEL (defaults to gemini-3.6-flash).
    Temperature is set low (0.2) for factual, consistent outputs.
    """
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.2
    )


def split_transcript(transcript: str) -> list:
    """Splits a long transcript into chunks for processing.

    Uses RecursiveCharacterTextSplitter to maintain semantic coherence
    at chunk boundaries via a 200-character overlap.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )
    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    """Generates a professional meeting summary using map-reduce.

    Steps:
      1. Split the transcript into chunks
      2. Summarize each chunk individually
      3. Combine partial summaries into a final bullet-point summary
    """
    llm = get_llm()

    # --- Map phase: summarize each chunk ---
    map_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize this portion of an audio transcript concisely."),
        ("human", "{text}"),
    ])
    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)
    chunk_summaries = [map_chain.invoke({"text": chunk}) for chunk in chunks]
    combined = "\n\n".join(chunk_summaries)

    # --- Reduce phase: combine into final summary ---
    reduce_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert meeting summarizer. Combine these partial summaries "
            "into one final professional summary in bullet points."
        ),
        ("human", "{text}"),
    ])
    reduce_chain = reduce_prompt | llm | StrOutputParser()

    return reduce_chain.invoke({"text": combined})


def generate_title(transcript: str) -> str:
    """Generates a concise, professional meeting title from the transcript.

    Returns only the title string (max 8 words).
    """
    llm = get_llm()

    title_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Based on the meeting transcript, generate a professional meeting title "
            "(max 8 words). Only return the title, nothing else."
        ),
        ("human", "{text}"),
    ])

    title_chain = title_prompt | llm | StrOutputParser()

    return title_chain.invoke({"text": transcript})
