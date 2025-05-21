## 📰 News Sentiment Classifier
A web application that classifies the sentiment of news headlines or paragraphs into Negative, Neutral, or Positive using an LSTM-based deep learning model built with TensorFlow and deployed with Streamlit.

## 🔗 Model Download
📥 [Download model.keras from Hugging Face](https://huggingface.co/destianiic/lstm-model/blob/main/model.keras)

📥 [Download tokenizer.pkl](https://huggingface.co/destianiic/lstm-model/blob/main/tokenizer.pkl)Download tokenizer.pkl from Hugging Face

## 🚀 Predict News Sentiment
You can try the app directly from your browser:

🔗 [Launch the App](https://lstm-text-classifier.streamlit.app/)

## Steps:
- Enter the text of a news article, headline, or paragraph into the input box.
- Click the Predict button.
- The model will output the predicted sentiment along with the confidence score.

## 🧠 Model Information
### Algorithm: Long Short-Term Memory (LSTM)
### Dataset: 1500 labeled news samples (500 for each class: Negative, Neutral, Positive)
### Text Preprocessing:
  - Lowercasing
  - Removing special characters
  - Stopword removal (Indonesian, using Sastrawi)
  - Tokenization and sequence padding
### Tokenizer: Trained and saved using Tokenizer from tensorflow.keras.preprocessing.text
### Model Architecture:
  - Embedding Layer
  - SpatialDropout1D
  - LSTM Layer
  - Dense Output Layer with Softmax
### Training:
  - Epochs: 5
  - Batch size: 64
  - Validation split: 0.2
  - Optimizer: Adam

## 🛠 How to Run Locally

### Clone the repository
git clone https://github.com/destianiic/lstm-text-classifier.git
cd lstm-text-classifier

### Install dependencies
pip install -r requirements.txt

### Run the Streamlit app
streamlit run app.py
Ensure you place the downloaded model.keras and tokenizer.pkl files in the model/ directory.

## 📁 Project Structure
├── app.py                 # Streamlit app
├── model/
│   ├── model.keras        # Trained LSTM model
│   └── tokenizer.pkl      # Tokenizer used during training
├── requirements.txt       # List of Python dependencies
└── README.md              # Project documentation


## 👤 Author
#### Destiani I. C.
#### GitHub: @destianiic
