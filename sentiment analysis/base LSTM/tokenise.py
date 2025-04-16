import pickle
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences #type: ignore
import numpy as np
import pandas as pd
df = pd.read_csv(r"sentiment analysis\cleaned_IMDB_Dataset.csv")

def tokenise(df,save_tokenizer=False):
    #tokenise the text
    tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")  # Use 20k most common words
    tokenizer.fit_on_texts(df["review"])  # Fit on the reviews column

    # Convert reviews to sequences
    sequences = tokenizer.texts_to_sequences(df["review"])
    
    # padding
    maxlen = 300  # Adjust as needed(max length of the padded reviews)

    # Apply padding
    padded_sequences = pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')

    #preparing labels now
    labels = np.where(df['sentiment']=='positive',1,0)
    print("done successfully!")
    if save_tokenizer:
        with open("tokenizer.pkl", "wb") as f:
            pickle.dump(tokenizer, f)
        print("Tokenizer saved successfully!")
        '''
        to use later
        with open("/content/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
        '''

    print("Tokenization done successfully!")
    return padded_sequences, labels, tokenizer
