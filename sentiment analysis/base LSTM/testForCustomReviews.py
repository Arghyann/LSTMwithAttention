#test with custom text

import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load trained model
model = tf.keras.models.load_model(r"sentiment analysis\saved files\sentiment_model.keras")

# Load tokenizer
with open(r"sentiment analysis\saved files\tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


def predict_sentiment(review):
    # Tokenize and pad the input review
    sequence = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequence, maxlen=300, padding='post', truncating='post')

    # Make prediction
    prediction = model.predict(padded_sequence)[0][0]

    # Interpret result
    sentiment = "Positive 😊" if prediction > 0.5 else "Negative 😞"
    return sentiment, prediction

while True:
    text = input("Enter a movie review (or 'exit' to quit): ")
    if text.lower() == "exit":
        break
    sentiment, score = predict_sentiment(text)
    print(f"Sentiment: {sentiment} (Confidence: {score:.4f})")