import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = load_model('/Users/destianiic/Documents/LSTM/model.h5')
with open('/Users/destianiic/Documents/LSTM/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

MAX_SEQUENCE_LENGTH = 250

# Define your class labels (adjust the order if needed)
class_labels = ['Negative', 'Neutral', 'Positive']

# Streamlit UI
st.title("🧠 LSTM Text Classifier")

user_input = st.text_area("Enter your news text here:")

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