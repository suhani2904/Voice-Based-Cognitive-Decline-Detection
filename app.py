import streamlit as st
import numpy as np
import joblib
import tempfile
import soundfile as sf
import time
import warnings
warnings.filterwarnings("ignore")

from acoustic_features import acoustic_features
from linguistic_features import transcribe_and_embed
from Segments import segment_audio
from audio_recorder_streamlit import audio_recorder

# Load trained model
start = time.time()
@st.cache_resource
def load_model():
    return joblib.load("pipeline.pkl")   # your trained model

model = load_model()

st.title("🧠 Voice-Based Dementia Detection")
st.write("Upload or record a voice sample to analyze acoustic and linguistic features.")

# Option 1: Upload File 
uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

# Option 2: Record Audio 
st.write("🎤 Or record your voice:")
audio_bytes = audio_recorder(pause_threshold=2.0, sample_rate=16000)

# Decide source
if uploaded_file is not None:
    source = uploaded_file.read()
    st.audio(uploaded_file, format="audio/wav")
elif audio_bytes is not None:
    source = audio_bytes
    st.audio(source, format="audio/wav")
else:
    source = None

if source:
    # Save audio to a temporary file
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_file.write(source)
    tmp_file.flush()

    segments = segment_audio(tmp_file.name, segment_len=6, overlap=3)

    all_preds = []
    for seg in segments:
        acoustic = acoustic_features(seg)   
        linguistic = transcribe_and_embed(seg)  

        if linguistic is None or acoustic is None:
            continue

        # Combine
        features = np.concatenate([acoustic, linguistic])
        features = features.reshape(1, -1)

        # Predict
        pred = model.predict(features)[0]
        all_preds.append(pred)

    if all_preds:
        print(all_preds)
        final_pred = int(np.round(np.mean(all_preds))) 
        print(final_pred) # average vote
        st.subheader("🩺 Prediction Result:")
        print(time.time() - start)
        if final_pred == 1:
            st.error("⚠️ Dementia Detected")
        else:
            st.success("✅ No Dementia Detected")
    else:
        st.warning("Not valid")

