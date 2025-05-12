import streamlit as st
import os
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from audio_recorder_streamlit import audio_recorder
import joblib

@st.cache_resource
def load_models():
    return {
        'model': joblib.load('my_model.joblib'),
        'scaler': joblib.load('my_scaler.joblib'),
        'pca': joblib.load('my_pca.joblib'),
    }

@st.cache_data
def load_training_data():
    df = pd.read_csv('output.csv')
    X_train = df.drop(columns=['label'], errors='ignore')
    scaler = load_models()['scaler']
    X_train_scaled = scaler.transform(X_train)
    pca = load_models()['pca']
    return pca.transform(X_train_scaled)

from extract_features import extract_features

# Set page config
st.set_page_config(page_title="Cognitive Decline Detection", layout="wide")
st.title('MemoTag: Cognitive Decline Detection')

# Create sidebar for app information
with st.sidebar:
    st.header("About This Tool")
    st.write("""
    This tool analyzes speech patterns to detect potential signs of cognitive decline.
    
    **Features analyzed:**
    - Pauses per sentence
    - Hesitation markers (uh, um)
    - Word recall issues
    - Speech rate and pitch variability
    - Naming & word association
    """)

# Main app interface
st.subheader("Upload an Audio File")
uploaded_file = st.file_uploader("Choose an audio file", type=['wav'])

# Provide test recording option
audio_input = st.radio("Audio Source", ["Upload File", "Record Audio"])
models = load_models()
X_train_pca = load_training_data()

if audio_input == "Record Audio":
    audio_data = audio_recorder()
    if audio_data:
        # Save the recorded audio to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_data)
            uploaded_file = tmp_file.name

# Process uploaded or recorded audio
if uploaded_file:
    with st.spinner("Processing audio..."):
        # Save uploaded file to temp location if it's not already a path
        if isinstance(uploaded_file, str):
            audio_path = uploaded_file
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                audio_path = tmp_file.name
        
        # extract features and transcript
        transcript , features = extract_features(audio_path)
        # convert the features dict to dataframe
        new_sample = pd.DataFrame([features])
        # scaling the user input
        scaled_sample = models['scaler'].transform(new_sample)
        # performing pca to extract two important features
        user_pca = models['pca'].transform(scaled_sample)
        # use kmeans model to predict risk
        y_pred = models['model'].predict(user_pca)

        # as y_pred is in numbers(0 or 1 or 2)
        if y_pred == 0:
            risk_level = "Low Risk"
            st.header("LOW RISK")
        elif y_pred == 2:
            risk_level = "Moderate risk"
            st.header("MODERATE RISK")
        else:
            risk_level = "High Risk"
            st.header("HIGH RISK")


        # Create two columns
        col1, col2 = st.columns([1, 2])  

        # First column is for the table
        with col1:
            st.subheader("Extracted Speech Features")
            # Transpose the dataframe to show features as rows
            features_table = new_sample.T.reset_index()
            features_table.columns = ['Feature', 'Value']
            st.dataframe(features_table.set_index('Feature'), 
                        use_container_width=True)
            

        with col2:
            # Create and display the chart
            fig, ax = plt.subplots(figsize=(8,6))
            sns.scatterplot(x=X_train_pca[:,0], y=X_train_pca[:,1], label='Training Data', color='blue', alpha=0.7, ax=ax)
            ax.scatter(user_pca[0,0], user_pca[0,1], color='red', s=200, marker='*', label='User Input')
            ax.set_xlabel('PCA Component 1')
            ax.set_ylabel('PCA Component 2')
            ax.set_title('Speech Pattern Analysis')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

        # showing transcript
        st.subheader("Transcript")
        st.write(transcript)

        # Interpretation
        st.subheader("Clinical Interpretation")
        if risk_level == "High Risk" or risk_level == "Moderate risk":
            st.warning("""
            This sample shows speech patterns that may indicate cognitive stress or early decline.
            Key indicators include hesitations, pauses, and word recall difficulties that are
            outside typical ranges. Please note this is not a clinical diagnosis.
            """)
        else:
            st.success("""
            This sample shows speech patterns within normal ranges. No significant
            indicators of cognitive decline were detected. Continue regular monitoring.
            """)

        
