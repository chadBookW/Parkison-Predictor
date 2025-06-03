import streamlit as st
import pandas as pd
import numpy as np
from model import make_prediction
import librosa

# Function to extract features from the uploaded audio file
def extract_features_from_audio(audio_file):
    # Load the audio file and resample to 16000 Hz
    audio, sr = librosa.load(audio_file, sr=16000)

    # Extract features
    features = {
        'MDVP:Fo(Hz)': np.mean(librosa.feature.zero_crossing_rate(audio)),
        'MDVP:Fhi(Hz)': np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)),
        'MDVP:Flo(Hz)': np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr)),
        'MDVP:Jitter(%)': np.mean(librosa.feature.rms(y=audio)),
        'MDVP:Jitter(Abs)': np.mean(librosa.feature.tempogram(y=audio)),
        'MDVP:RAP': np.mean(librosa.feature.chroma_stft(y=audio, sr=sr)),
        'MDVP:PPQ': np.mean(librosa.feature.chroma_cqt(y=audio, sr=sr)),
        'Jitter:DDP': np.mean(librosa.feature.chroma_cens(y=audio, sr=sr)),
        'MDVP:Shimmer': np.mean(librosa.feature.mfcc(y=audio, sr=sr)),
        'MDVP:Shimmer(dB)': np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)),
        'Shimmer:APQ3': np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr)),
        'Shimmer:APQ5': np.mean(librosa.feature.spectral_flatness(y=audio)),  # Removed sr argument
        'MDVP:APQ': np.mean(librosa.feature.tonnetz(y=audio)),
        'Shimmer:DDA': np.mean(librosa.feature.poly_features(y=audio)),
        'NHR': 0.05,  # Placeholder for now
        'HNR': 21.0,  # Placeholder for now
        'RPDE': 0.5,  # Placeholder for now
        'DFA': 0.7,  # Placeholder for now
        'spread1': -4.0,  # Placeholder for now
        'spread2': 0.1,  # Placeholder for now
        'D2': 1.5,  # Placeholder for now
        'PPE': 0.2   # Placeholder for now
    }

    return pd.DataFrame([features])

#sidebar
def user_input_features():
    st.sidebar.header("Input Voice Measurements")

    #sliders for manual feature input
    MDVP_Fo = st.sidebar.slider('MDVP:Fo(Hz)', 88.0, 260.0, 154.0)
    MDVP_Fhi = st.sidebar.slider('MDVP:Fhi(Hz)', 100.0, 600.0, 197.0)
    MDVP_Flo = st.sidebar.slider('MDVP:Flo(Hz)', 65.0, 239.0, 120.0)
    MDVP_Jitter = st.sidebar.slider('MDVP:Jitter(%)', 0.0, 0.02, 0.01)
    MDVP_Jitter_Abs = st.sidebar.slider('MDVP:Jitter(Abs)', 0.0, 0.0003, 0.0001)
    MDVP_RAP = st.sidebar.slider('MDVP:RAP', 0.0, 0.02, 0.01)
    MDVP_PPQ = st.sidebar.slider('MDVP:PPQ', 0.0, 0.02, 0.01)
    Jitter_DDP = st.sidebar.slider('Jitter:DDP', 0.0, 0.04, 0.02)
    MDVP_Shimmer = st.sidebar.slider('MDVP:Shimmer', 0.0, 0.1, 0.03)
    MDVP_Shimmer_dB = st.sidebar.slider('MDVP:Shimmer(dB)', 0.0, 1.0, 0.35)
    Shimmer_APQ3 = st.sidebar.slider('Shimmer:APQ3', 0.0, 1.0, 0.3)
    Shimmer_APQ5 = st.sidebar.slider('Shimmer:APQ5', 0.0, 1.0, 0.3)
    MDVP_APQ = st.sidebar.slider('MDVP:APQ', 0.0, 1.0, 0.5)
    Shimmer_DDA = st.sidebar.slider('Shimmer:DDA', 0.0, 1.0, 0.3)
    NHR = st.sidebar.slider('NHR', 0.0, 0.5, 0.05)
    HNR = st.sidebar.slider('HNR', 0.0, 40.0, 21.0)
    RPDE = st.sidebar.slider('RPDE', 0.2, 0.7, 0.5)
    DFA = st.sidebar.slider('DFA', 0.5, 1.0, 0.7)
    spread1 = st.sidebar.slider('spread1', -7.0, 0.0, -4.0)
    spread2 = st.sidebar.slider('spread2', 0.0, 0.5, 0.1)
    D2 = st.sidebar.slider('D2', 0.0, 3.0, 1.5)
    PPE = st.sidebar.slider('PPE', 0.0, 0.5, 0.2)

    features = {
        'MDVP:Fo(Hz)': MDVP_Fo,
        'MDVP:Fhi(Hz)': MDVP_Fhi,
        'MDVP:Flo(Hz)': MDVP_Flo,
        'MDVP:Jitter(%)': MDVP_Jitter,
        'MDVP:Jitter(Abs)': MDVP_Jitter_Abs,
        'MDVP:RAP': MDVP_RAP,
        'MDVP:PPQ': MDVP_PPQ,
        'Jitter:DDP': Jitter_DDP,
        'MDVP:Shimmer': MDVP_Shimmer,
        'MDVP:Shimmer(dB)': MDVP_Shimmer_dB,
        'Shimmer:APQ3': Shimmer_APQ3,
        'Shimmer:APQ5': Shimmer_APQ5,
        'MDVP:APQ': MDVP_APQ,
        'Shimmer:DDA': Shimmer_DDA,
        'NHR': NHR,
        'HNR': HNR,
        'RPDE': RPDE,
        'DFA': DFA,
        'spread1': spread1,
        'spread2': spread2,
        'D2': D2,
        'PPE': PPE
    }

    return pd.DataFrame([features])

#app layout
st.title("Parkinson's Disease Early Detection")

st.write("""
This app predicts whether a person has Parkinson's Disease based on their voice measurements.
You can either use the sliders to input the measurements or upload an audio file.
""")

# option to upload an audio file
audio_file = st.file_uploader("Upload an audio file (WAV format)", type=["wav"])

# process uploaded audio file
if audio_file is not None:
    input_df = extract_features_from_audio(audio_file)
else:
    # User inputs from the sidebar if no audio file is uploaded
    input_df = user_input_features()

# Display user inputs
st.subheader("User Inputs")
st.write(input_df)

# Make prediction
prediction, prediction_proba = make_prediction(input_df)

# Display prediction result
st.subheader("Prediction")
if prediction == 1:
    st.write("The model predicts the person has Parkinson's Disease.")
else:
    st.write("The model predicts the person is healthy.")

# Display customized prediction probabilities
st.subheader("Prediction Probability")
st.write(f"Probability of having Parkinson's Disease: **{prediction_proba[0][1]:.2f}**")
st.write(f"Probability of being healthy: **{prediction_proba[0][0]:.2f}**")