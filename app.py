import streamlit as st
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model
import joblib
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
text_input = st.text_area("Enter a news headline or paragraph:", height=150)

if st.button("Predict"):
    if user_input.strip():
        # Preprocess input
        seq = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)

        # Predict
        prediction = model.predict(padded)
        class_idx = np.argmax(prediction)
        class_label = class_labels[class_idx]

        confidence = prediction[0][class_idx]

        st.success(f"Predicted sentiment: **{class_label}** (Confidence: {confidence:.2f})")
    else:
        st.warning("Please enter some text.")
