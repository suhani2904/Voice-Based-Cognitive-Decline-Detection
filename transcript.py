import assemblyai as aai
import os
from dotenv import load_dotenv

load_dotenv()

aai.settings.api_key = os.getenv("ASSEMBLE_AI_API")

config = aai.TranscriptionConfig(
  disfluencies=True,
  punctuate=True,
  format_text=True,
  word_boost=["uh" , "I mean" , "um" ],
  boost_param="high",
  )

def convert_audio_to_text(audio_path):
    transcript = aai.Transcriber(config=config).transcribe(audio_path)

    if transcript.error:
        print(f"Error: {transcript.error}")
        return None
    
    return transcript.text , transcript.words






    



