# 🎙️ Voice-Based Cognitive Decline Detection

## 📌 Overview
Voice-based cognitive decline detection is an AI-driven healthcare project that analyzes speech patterns to identify early signs of cognitive impairment such as dementia or Alzheimer’s disease.

This project uses **acoustic and linguistic features extracted from speech recordings** and applies machine learning models to detect potential cognitive decline.

Early detection can help healthcare professionals intervene sooner and improve patient outcomes.

---

## 🚀 Features
- Speech-based cognitive health analysis
- Acoustic feature extraction using **OpenSMILE**
- Audio processing using **librosa**
- Linguistic feature extraction using **RoBERTa**
- Multiple ML models for classification
- Model comparison and optimization
- Focus on reducing **false negatives** for reliable detection

---

## 🛠️ Tech Stack

**Programming Language**
- Python

**Libraries & Frameworks**
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- Librosa
- OpenSMILE
- HuggingFace Transformers (RoBERTa)
- Matplotlib / Seaborn

---

## 🔬 Project Pipeline

1. **Data Collection**
   - Created a dataset of speech recordings.

2. **Preprocessing**
   - Audio normalization
   - Noise removal

3. **Feature Extraction**
   - Acoustic features (MFCCs, pitch, spectral features)
   - Linguistic features from speech transcripts

4. **Model Training**
   - Support Vector Machine (SVM)
   - Random Forest
   - XGBoost

5. **Model Evaluation**
   - Accuracy
   - Precision
   - Recall
   - F1 Score

---

## 📊 Model Performance

| Model | Accuracy |
|------|---------|
| SVM | ~90% |
| Random Forest | ~92% |
| XGBoost | **~94%** |

XGBoost achieved the best performance for detecting cognitive decline.

---
