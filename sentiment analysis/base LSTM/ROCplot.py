import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from tokenise import tokenise  # Assuming same tokenizer file

# Load model
model = tf.keras.models.load_model("sentiment_model.keras")

# Load and tokenize data
df = pd.read_csv(r"sentiment analysis\cleaned_IMDB_Dataset.csv")
padded_sequences, labels, _ = tokenise(df, save_tokenizer=False)

# Test split (same as before)
from sklearn.model_selection import train_test_split
_, X_test, _, y_test = train_test_split(
    padded_sequences, labels, test_size=0.2, random_state=42
)

# Predict probabilities
y_pred_proba = model.predict(X_test).ravel()

# Compute ROC curve and AUC
from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})', color='darkorange')
plt.plot([0, 1], [0, 1], 'k--', label='Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('roc_curve.png')
plt.show()
