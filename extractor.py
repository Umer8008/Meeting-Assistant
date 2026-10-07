"""
Meeting content extraction module.

Provides focused LLM-based extraction functions for specific meeting elements:
  - Key points
  - Important decisions
  - Questions raised
  - Action items / tasks

Each function uses a dedicated system prompt to ensure precise extraction.
Uses the same Gemini LLM instance as the summarization module.
"""

from core.summarize import get_llm, split_transcript
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def _extract(transcript: str, system_prompt: str) -> str:
    """Internal helper — runs a single extraction task against the transcript.

    For long transcripts, processes in chunks and merges results to stay
    within LLM context limits.
    """
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}"),
    ])
    chain = prompt | llm | StrOutputParser()

    # If transcript is short enough, process directly
    if len(transcript) < 3000:
        return chain.invoke({"text": transcript})

    # For long transcripts, extract from each chunk then consolidate
    chunks = split_transcript(transcript)
    partial_results = [chain.invoke({"text": chunk}) for chunk in chunks]
    combined = "\n\n".join(partial_results)

    # Consolidation pass — merge partial extractions into a clean list
    merge_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            f"You received partial extraction results from a long meeting transcript. "
            f"Merge and deduplicate them into a single clean list. "
            f"Original task: {system_prompt}"
        ),
        ("human", "{text}"),
    ])
    merge_chain = merge_prompt | llm | StrOutputParser()
    return merge_chain.invoke({"text": combined})


def extract_key_points(transcript: str) -> str:
    """Extracts the most important points discussed in the meeting.

    Returns a numbered or bulleted list of key points.
    """
    return _extract(
        transcript,
        "You are an expert meeting analyst. Extract the most important key points "
        "discussed in this meeting transcript. Present them as a clean numbered list. "
        "Focus on substantive points, not procedural details."
    )


def extract_decisions(transcript: str) -> str:
    """Extracts important decisions made during the meeting.

    Returns a numbered list of decisions with brief context.
    """
    return _extract(
        transcript,
        "You are an expert meeting analyst. Extract all important decisions made "
        "during this meeting. Present each decision as a numbered item with brief "
        "context about what was decided and why. Only include actual decisions, "
        "not suggestions or open questions."
    )


def extract_questions(transcript: str) -> str:
    """Extracts questions raised during the meeting.

    Returns a numbered list of questions, including both answered
    and unanswered ones.
    """
    return _extract(
        transcript,
        "You are an expert meeting analyst. Extract all notable questions raised "
        "during this meeting. Present them as a numbered list. Include both "
        "questions that were answered and those left open. For answered questions, "
        "briefly note the answer if provided."
    )


def extract_action_items(transcript: str) -> str:
    """Extracts action items and tasks assigned during the meeting.

    Returns structured action items. If responsible persons or deadlines
    are mentioned in the transcript, includes them; otherwise lists
    only the tasks themselves.
    """
    return _extract(
        transcript,
        "You are an expert meeting analyst. Extract all action items and tasks "
        "assigned during this meeting. For each action item, include:\n"
        "- Task description\n"
        "- Responsible person (if mentioned)\n"
        "- Deadline (if mentioned)\n"
        "Present them as a numbered list. Only include information actually "
        "stated in the transcript — do not invent details."
    )
