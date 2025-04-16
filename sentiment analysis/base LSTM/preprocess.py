import pandas as pd
import re
import string
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download stopwords if not already present
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_text(text):
    """Cleans text by removing HTML tags, punctuation, and stopwords."""
    text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML tags
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return " ".join(words)

# Load dataset
df = pd.read_csv("IMDB Dataset.csv", names=["review", "sentiment"])

# Preprocess text
df["cleaned_review"] = df["review"].apply(clean_text)

# Tokenization
max_words = 10000  # Vocabulary size
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(df["cleaned_review"])
sequences = tokenizer.texts_to_sequences(df["cleaned_review"])

# Padding sequences
max_length = 200  # Max length of a review
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding="post", truncating="post")

# Save preprocessed data
df.to_csv("cleaned_IMDB_Dataset.csv", index=False)
print("Preprocessing complete! Data saved as 'cleaned_IMDB_Dataset.csv'.")