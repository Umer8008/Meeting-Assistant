
from utils.audio_processor import process_input
from core.transcribe import transcribe_all

source = "https://youtu.be/FN6iUHwieho?si=CyO9JtnLbPdLWU5M"

chunks = process_input(source)

print(transcribe_all(chunks))

#.venv\Scripts\streamlit.exe run app.py

