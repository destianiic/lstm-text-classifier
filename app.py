import streamlit as st
from huggingface_hub import hf_hub_download
import joblib
import numpy as np

# Constants
class_labels = ['Negative', 'Neutral', 'Positive']

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model_path = hf_hub_download(repo_id="destianiic/lstm-model", filename="model.pkl")
    tokenizer_path = hf_hub_download(repo_id="destianiic/lstm-model", filename="tokenizer.pkl")
    model = joblib.load(model_path)
    tokenizer = joblib.load(tokenizer_path)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# Streamlit UI
st.title("📰 News Sentiment Classifier")
text_input = st.text_area("Enter a news headline or paragraph:", height=150)

if st.button("Predict"):
    if text_input.strip():
        seq = tokenizer.texts_to_sequences([text_input])
        padded = tokenizer.pad_sequences(seq, maxlen=250)  # Adjust this based on your tokenizer

        prediction = model.predict_proba(padded)
        class_idx = np.argmax(prediction)
        class_label = class_labels[class_idx]
        confidence = prediction[0][class_idx]

        st.success(f"Predicted sentiment: **{class_label}** (Confidence: {confidence:.2f})")
    else:
        st.warning("Please enter some text.")
