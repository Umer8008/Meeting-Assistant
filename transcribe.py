import os
import torch
import whisper

# Configuration from Environment Variables
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

# Global Lazy-Loaded Model Instance
_model = None


# ==========================================
# STANDARD WHISPER MODEL
# ==========================================

def load_model():
    """Lazy loads and caches the multilingual Whisper model."""
    global _model
    if _model is None:
        print("Loading Whisper model....")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        _model = whisper.load_model(
            WHISPER_MODEL,
            device=device
        )

        print("Whisper model loaded successfully.")

    return _model


# ==========================================
# LANGUAGE DETECTION
# ==========================================

def detect_language(chunk_path: str) -> str:
    """Detects the primary spoken language in an audio segment."""
    model = load_model()

    print("Detecting audio language....")

    audio = whisper.load_audio(chunk_path)

    # Whisper language detection uses a 30-second audio window
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(
        audio,
        n_mels=model.dims.n_mels
    ).to(model.device)

    _, probabilities = model.detect_language(mel)

    language = max(
        probabilities,
        key=probabilities.get
    )

    print(f"Detected language: {language}")

    return language


# ==========================================
# TRANSCRIBE ONE CHUNK
# ==========================================

def transcribe_chunk(
    chunk_path: str,
    translate: bool = False
) -> str:
    """Transcribes or translates a single audio chunk."""
    model = load_model()

    task = "translate" if translate else "transcribe"

    print(
        "Processing chunk → "
        f"{'Translation to English' if translate else 'Original language transcription'}"
    )

    result = model.transcribe(
        chunk_path,
        task=task,
        fp16=torch.cuda.is_available()
    )

    return result["text"]


# ==========================================
# TRANSCRIBE ALL CHUNKS
# ==========================================

def transcribe_all(
    chunks: list[str],
    translate: bool = False,
    mode: str = None,
    progress_callback=None
) -> dict:
    """Processes all audio chunks and returns the final transcript.

    Args:
        chunks: List of file paths to audio chunks.
        translate: Whether to use Whisper's translate task (English output).
        mode: Transcription mode — "1" (original), "2" (English), "3" (Urdu).
              When None, falls back to interactive CLI prompt for backward
              compatibility with scripts like test.py.
        progress_callback: Optional callable(current_chunk, total_chunks, message)
                          for GUI progress updates.

    Returns:
        dict with keys: "transcript", "language", "mode_label"
    """
    if not chunks:
        return {"transcript": "", "language": "unknown", "mode_label": "N/A"}

    # Detect the language from the first chunk
    detected_language = detect_language(chunks[0])

    print(f"Detected language: {detected_language}")

    # If no mode was passed (CLI usage), prompt the user interactively
    if mode is None:
        choice = input(
            "\nChoose output:\n"
            "1. Transcribe in original language\n"
            "2. Translate to English\n"
            "3. Translate to Urdu\n"
            "Enter your choice: "
        ).strip()
    else:
        choice = str(mode)

    mode_label = "Original Language"

    # Original language transcription
    if choice == "1":
        translate = False
        mode_label = "Original Language"

    # Whisper's built-in speech translation
    elif choice == "2":
        translate = True
        mode_label = "English Translation"

    # Urdu translation will be handled by the separate translation model later
    elif choice == "3":
        translate = False
        mode_label = "Urdu (pending translation model)"
        print(
            "Urdu translation will be handled by the translation model later."
        )

    else:
        print("Invalid choice. Transcribing in original language.")
        translate = False

    full_transcript = []

    for i, chunk in enumerate(chunks):

        print(f"Transcribing Chunk {i + 1} of {len(chunks)}")

        # Notify the GUI of progress if a callback is provided
        if progress_callback:
            progress_callback(i + 1, len(chunks), f"Transcribing chunk {i + 1} of {len(chunks)}...")

        text = transcribe_chunk(
            chunk,
            translate=translate
        )

        full_transcript.append(text)

    print("Transcription completed")

    return {
        "transcript": "\n".join(full_transcript),
        "language": detected_language,
        "mode_label": mode_label
    }