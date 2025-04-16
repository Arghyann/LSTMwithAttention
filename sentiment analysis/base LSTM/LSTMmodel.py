from tokenise import tokenise
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.regularizers import l2

# Load your dataset
df = pd.read_csv(r"sentiment analysis\cleaned_IMDB_Dataset.csv")
padded_sequences, labels, tokenizer = tokenise(df, save_tokenizer=True)

X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# Configs
embedding_dim = 100  # Match the GloVe file (100d)
lstm_units = 64
vocab_size = 20000
maxlen = 300

# Load GloVe vectors
embedding_index = {}
with open(r"sentiment analysis\GloVe\glove.6B\glove.6B.100d.txt", encoding="utf8") as f:
    for line in f:
        values = line.strip().split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = vector

# Create embedding matrix
word_index = tokenizer.word_index
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in word_index.items():
    if i < vocab_size:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        # If not found in GloVe, stays 0

# Build the model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim,
              weights=[embedding_matrix], input_length=maxlen, trainable=False),  # Freeze embeddings
    LSTM(lstm_units, return_sequences=True),
    Dropout(0.2),
    LSTM(lstm_units),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

# Train
batch_size = 64
epochs = 10

history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    batch_size=batch_size, epochs=epochs, verbose=1)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")



# Save model
model.save("sentiment_model.keras")
