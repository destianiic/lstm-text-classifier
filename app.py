import streamlit as st
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import numpy as np

# Constants
MAX_SEQUENCE_LENGTH = 250
class_labels = ['Negative', 'Neutral', 'Positive']

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model_path = hf_hub_download(repo_id="destianiic/lstm-model", filename="model.h5")
    tokenizer_path = hf_hub_download(repo_id="destianiic/lstm-model", filename="tokenizer.pkl")
    model = load_model(model_path)
    tokenizer = joblib.load(tokenizer_path)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# Streamlit UI
st.title("📰 News Sentiment Classifier")
st.markdown("Classify the sentiment of a news headline or paragraph as **Negative**, **Neutral**, or **Positive**.")

text_input = st.text_area("Enter text:", height=150)

if st.button("Predict"):
    if text_input.strip():
        # Preprocess input
        seq = tokenizer.texts_to_sequences([text_input])
        padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)

        # Predict
        prediction = model.predict(padded)
        class_idx = np.argmax(prediction)
        class_label = class_labels[class_idx]
        confidence = prediction[0][class_idx]

        st.success(f"Predicted sentiment: **{class_label}** (Confidence: {confidence:.2f})")
    else:
        st.warning("Please enter some text.")
