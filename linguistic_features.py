import tempfile
import soundfile as sf
from sentence_transformers import SentenceTransformer
import assemblyai as aai
import os
from dotenv import load_dotenv


load_dotenv()
model = SentenceTransformer("sentence-transformers/all-roberta-large-v1")
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")


def transcribe_and_embed(segment, sr=16000):
    """Takes a NumPy segment → transcribe with AssemblyAI → return embeddings"""
    
    # Save segment to a temp wav file
    tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp_file.name, segment, sr)

    # Transcribe
    transcript = aai.Transcriber().transcribe(tmp_file.name)


    if transcript.status == "error":
        raise RuntimeError(f"Transcription failed: {transcript.error}")

    text = transcript.text.lower().strip()

    if not text:  # if transcript is empty
        return None

    # Convert transcript to embedding
    embedding = model.encode(text)
    return embedding
