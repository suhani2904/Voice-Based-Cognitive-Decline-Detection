# Voice-Based-Cognitive-Decline-Detection

## Project Overview

This project demonstrates a proof-of-concept pipeline for **early detection of cognitive stress or decline** using analysis of raw voice data. The system leverages both **acoustic** and **linguistic** features extracted from speech, applying unsupervised machine learning to identify abnormal patterns that may indicate mild cognitive impairment (MCI) or early-stage dementia.

---

## 🚀 Features

- **🔊 Speech-to-text transcription:** Uses Facebook's Wav2Vec2 for accurate, offline audio transcription.
- **🧠 Cognitive speech markers:** Extracts features such as hesitations, silent pauses, speech rate, pitch variability, and word recall issues.
- **🤖 Unsupervised anomaly & cluster detection:** Applies Isolation Forest and K-Means to flag speech patterns potentially indicative of cognitive decline.
- **📉 PCA-based visualization:** Projects feature space into 2D for cluster and anomaly visualization.
- **🌐 Interactive web interface:** User-friendly Streamlit app for uploading/recording audio, feature inspection, and risk assessment.

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **Libraries:**
  - assemblyai – (for optional ASR/transcription, if used)
  - 
  - dotenv – Environment variable management

  - librosa – Audio processing and feature extraction
    
  - pandas – Data manipulation and analysis
    
  - streamlit – Interactive web app interface
    
  - matplotlib, seaborn – Data visualization
    
  - audio_recorder_streamlit – In-browser audio recording
    
  - numpy – Numerical computing (pinned to <2.0.0 for compatibility)
    
  - spacy – Natural Language Processing (NLP)
    
  - en_core_web_sm – spaCy English language model

---
## 📂 How It Works

1. **Upload or record** an audio file (.wav).
2. **Transcription:** The audio is transcribed to text using Assemblyai.
3. **Feature extraction:** Acoustic and linguistic features are computed, including:
   - Hesitations and hedge words
   - Silent pause counts
   - Speech rate and pitch variability
   - Word repetition and recall issues
4. **Unsupervised analysis:** K-Means clustering and Isolation Forest flag outliers and cluster membership.
5. **Visualization:** PCA scatterplot shows your sample relative to the dataset.
6. **Interpretation:** The app provides a risk label ("High Risk"/"Low Risk"/"Moderate Risk) and a table of extracted features.


---

## ✅ Setup Instructions

1. **Clone the repository**

