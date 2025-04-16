

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from pathlib import Path # Import Path for potentially easier path handling

# --- Configuration Section ---
# Define the base directory where your data and model files are stored.
# Using Path('.').resolve() makes it relative to the script's location if run directly.
# Or you can specify an absolute path like: r'C:\Users\YourUser\Projects\sentiment_analysis'
BASE_DIR = Path(r'sentiment analysis/attention attempt').resolve() # Use forward slashes for better cross-platform compatibility initially

# Define filenames
WORD_TO_INDEX_FILENAME = r'D:\projects\python\sentiment-analysis\sentiment analysis\attention attempt\word_to_index.pkl'
EMBEDDING_MATRIX_FILENAME = r'D:\projects\python\sentiment-analysis\sentiment analysis\attention attempt\embedding_matrix.pt'
MODEL_WEIGHTS_FILENAME = r'D:\projects\python\sentiment-analysis\sentiment analysis\attention attempt\lstm_attention_model.pth'

# Construct full paths using os.path.join for robustness
WORD_TO_INDEX_PATH = os.path.join(BASE_DIR, WORD_TO_INDEX_FILENAME)
EMBEDDING_MATRIX_PATH = os.path.join(BASE_DIR, EMBEDDING_MATRIX_FILENAME)
MODEL_WEIGHTS_PATH = os.path.join(BASE_DIR, MODEL_WEIGHTS_FILENAME)

# --- Model Class Definition ---
class LSTMWithAttention(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, output_dim, dropout=0.5):
        super(LSTMWithAttention, self).__init__()

        # Embedding Layer
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(input_size=embedding_matrix.size(1),
                            hidden_size=hidden_dim,
                            num_layers=1,
                            bidirectional=True,
                            batch_first=True)

        # Attention Layer
        self.attn = nn.Linear(hidden_dim * 2, 1)

        # Dropout Layer
        self.dropout = nn.Dropout(dropout)

        # Fully Connected Layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def attention(self, lstm_out):
        """
        Attention mechanism to compute attention scores and weight LSTM outputs.
        """
        attn_weights = F.softmax(self.attn(lstm_out), dim=1)
        # Apply attention weights to LSTM outputs
        attn_output = torch.sum(attn_weights * lstm_out, dim=1)
        return attn_output, attn_weights

    def forward(self, x):
        # Get the embedded representation
        embedded = self.embedding(x)

        # Pass through LSTM layer
        lstm_out, (h_n, c_n) = self.lstm(embedded)

        # Apply attention mechanism - modified to return weights for visualization
        attn_out, attn_weights = self.attention(lstm_out)

        # Pass through dropout
        out = self.dropout(attn_out)

        # Pass through fully connected layer for classification
        output = self.fc(out)

        return output, attn_weights

# --- Streamlit App Setup ---
# Set page configuration
st.set_page_config(
    page_title="Sentiment Analysis with Attention",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for a nicer UI
st.markdown("""
    <style>
    /* --- Keep your existing rules above this line --- */
    .main { background-color: #f8f9fa; }
    .stApp { max-width: 1200px; margin: 0 auto; }
    h1, h2 { color: #1E3A8A; } /* General H1/H2 */
    /* ... other general styles ... */
    div.stButton > button:first-child { background-color: #28a745; color: white; /* ... */ }
    div.stButton > button:hover { background-color: #218838; }

    /* --- Result Box Styling --- */
    .result-box {
        padding: 15px 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        border: 1px solid #dee2e6;
    }
    /* Target elements *inside* the specific result box */
    .result-box h3 {
         margin-top: 0;
         margin-bottom: 8px;
         font-size: 1.4em;
         font-weight: 600;
    }
     .result-box p {
        font-size: 1.1em;
        margin-bottom: 0;
        line-height: 1.5;
    }

    /* Positive Result Box - SET DARK TEXT COLORS HERE */
    .result-box-positive {
        background-color: #e9f5ec; /* Light Green */
        border-left: 5px solid #28a745; /* Green */
    }
    .result-box-positive h3 { color: #155724; } /* Dark Green */
    .result-box-positive p { color: #34495e; } /* Dark Grayish Blue */
    .result-box-positive b { color: #155724; } /* Dark Green for bold */

    /* Negative Result Box - SET DARK TEXT COLORS HERE & FIX BG */
    .result-box-negative {
        background-color: #fef0f1; /* Corrected Light Red */
        border-left: 5px solid #dc3545; /* Red */
    }
    .result-box-negative h3 { color: #721c24; } /* Dark Red */
    .result-box-negative p { color: #34495e; } /* Dark Grayish Blue */
    .result-box-negative b { color: #721c24; } /* Dark Red for bold */

    </style>
    """, unsafe_allow_html=True)

# --- App Content ---
st.title("🎬 Movie Review Sentiment Analysis")
st.markdown("""
Analyze the sentiment of movie reviews using an LSTM model with attention.
Enter a review below to classify it as **Positive** or **Negative** and see which words the model focused on.
""")

# --- Helper Functions ---
@st.cache_resource
def load_model_and_data():
    """Loads the model, tokenizer, and sets up the device."""
    try:
        # Device configuration
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        st.info(f"Using device: {device}")

        # Load word_to_index mapping using the configured path
        if os.path.exists(WORD_TO_INDEX_PATH):
            with open(WORD_TO_INDEX_PATH, 'rb') as f:
                word_to_index = pickle.load(f)
            st.success("Loaded word_to_index vocabulary.")
        else:
            st.warning(f"word_to_index.pkl not found at '{WORD_TO_INDEX_PATH}'. Using a small demo vocabulary. Analysis quality will be limited.")
            # Fallback demo vocabulary
            word_to_index = {
                '<PAD>': 0, '<UNK>': 1, 'the': 2, 'a': 3, 'an': 4, 'movie': 5, 'film': 6, 'is': 7, 'was': 8, 'it': 9,
                'good': 10, 'great': 11, 'bad': 12, 'terrible': 13, 'not': 14, 'excellent': 15, 'poor': 16, 'boring': 17,
                'amazing': 18, 'awful': 19, 'best': 20, 'worst': 21, 'loved': 22, 'hated': 23, 'enjoyed': 24, 'disappointed': 25,
                'plot': 26, 'acting': 27, 'at': 28, 'all': 29, 'and': 30, 'even': 31, 'worse': 32
            }

        # Load embedding matrix using the configured path
        if os.path.exists(EMBEDDING_MATRIX_PATH):
            embedding_matrix = torch.load(EMBEDDING_MATRIX_PATH, map_location=device)
            st.success("Loaded pre-trained embedding matrix.")
        else:
            st.warning(f"embedding_matrix.pt not found at '{EMBEDDING_MATRIX_PATH}'. Using random embeddings. Analysis quality will be limited.")
            # Create a simple embedding matrix for demo
            vocab_size = len(word_to_index)
            embedding_dim = 100 # Should match model expectation if possible
            embedding_matrix = torch.rand(vocab_size, embedding_dim)

        # Initialize the model
        # Ensure these hyperparams match the *trained* model if loading weights
        model = LSTMWithAttention(
            embedding_matrix=embedding_matrix,
            hidden_dim=128,
            output_dim=2 # Assuming binary classification (Positive/Negative)
        )

        # Load saved model weights using the configured path
        if os.path.exists(MODEL_WEIGHTS_PATH):
            model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
            st.success("Loaded trained model weights.")
        else:
            st.error(f"Model weights ({MODEL_WEIGHTS_FILENAME}) not found at '{MODEL_WEIGHTS_PATH}'. Using an untrained model. Predictions will be random.")

        model.to(device)
        model.eval()  # Set to evaluation mode

        return model, word_to_index, device

    except Exception as e:
        st.error(f"Fatal error loading model or data: {str(e)}")
        st.error(f"Please ensure the following files exist at the configured paths:")
        st.error(f"- Vocabulary: {WORD_TO_INDEX_PATH}")
        st.error(f"- Embeddings: {EMBEDDING_MATRIX_PATH}")
        st.error(f"- Model Weights: {MODEL_WEIGHTS_PATH}")
        return None, None, None

def preprocess_review(review, word_to_index, max_length=100):
    """Process a text review into a tensor suitable for model input"""
    tokens = review.lower().split() # Simple whitespace tokenization
    tokenized = [word_to_index.get(token, word_to_index.get('<UNK>', 1)) for token in tokens] # Use <UNK> index

    # Truncate or Pad
    if len(tokenized) > max_length:
        tokenized = tokenized[:max_length]
    elif len(tokenized) < max_length:
        padding = [word_to_index.get('<PAD>', 0)] * (max_length - len(tokenized)) # Use <PAD> index
        tokenized = tokenized + padding

    review_tensor = torch.tensor(tokenized)
    return review_tensor.unsqueeze(0) # Add batch dimension

def visualize_attention(review_text, attn_weights):
    """Create a visualization of attention weights"""
    tokens = review_text.lower().split()
    weights = attn_weights.squeeze().detach().cpu().numpy()
    weights = weights[:len(tokens)] # Ignore padding weights

    if not tokens: # Handle empty input
        return plt.figure()

    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(4, len(tokens) * 0.3))) # Adjust height based on number of tokens

    # Sort by attention weight for better visualization
    valid_indices = np.arange(len(tokens))
    sorted_indices = sorted(valid_indices, key=lambda i: weights[i]) # Sort original indices based on weights
    sorted_tokens = [tokens[i] for i in sorted_indices]
    sorted_weights = [weights[i] for i in sorted_indices]

    # Create horizontal bar chart
    y_pos = np.arange(len(sorted_tokens))
    bars = ax.barh(y_pos, sorted_weights, align='center', height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_tokens, fontsize=10)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel('Attention Weight', fontsize=12)
    ax.set_title('Words Ranked by Model Attention', fontsize=14)
    ax.tick_params(axis='x', labelsize=10)

    # Add color gradient (e.g., cool to warm: low to high attention)
    norm = plt.Normalize(min(sorted_weights) if sorted_weights else 0, max(sorted_weights) if sorted_weights else 1)
    cmap = plt.get_cmap('viridis') # Or 'plasma', 'magma', 'cividis'

    for i, bar in enumerate(bars):
        bar.set_color(cmap(norm(sorted_weights[i])))

    plt.tight_layout()
    return fig

def create_highlighted_text(review_text, attn_weights):
    """Create HTML with highlighted text based on attention weights"""
    tokens = review_text.lower().split()
    weights = attn_weights.squeeze().detach().cpu().numpy()[:len(tokens)]

    if not tokens: # Handle empty input
        return ""

    # Normalize weights for coloring (0 to 1)
    max_weight = weights.max() if weights.size > 0 else 0
    min_weight = weights.min() if weights.size > 0 else 0
    # Avoid division by zero if all weights are the same
    if max_weight == min_weight:
         norm_weights = np.zeros_like(weights) + 0.5 # Assign medium intensity if all weights are equal
    elif max_weight > 0:
        # Scale weights to be more visually distinct, e.g., boost lower weights slightly
        norm_weights = (weights - min_weight) / (max_weight - min_weight)
        norm_weights = np.power(norm_weights, 0.7) # Apply gamma correction to enhance visibility
    else:
        norm_weights = np.zeros_like(weights) # All weights are zero or negative

    html_parts = []
    # Use a perceptually uniform colormap like Viridis (converted to RGB)
    cmap = plt.get_cmap('viridis')

    for token, weight_norm in zip(tokens, norm_weights):
        # Get RGBA color from colormap, use alpha for intensity
        rgba_color = cmap(weight_norm)
        # Create rgba string, ensure alpha is between 0.1 (min visibility) and 1.0
        alpha = max(0.1, weight_norm * 0.8 + 0.1) # Scale alpha: more intense for higher weights
        bg_color = f"rgba({int(rgba_color[0]*255)}, {int(rgba_color[1]*255)}, {int(rgba_color[2]*255)}, {alpha:.2f})"
        # Determine text color for contrast
        text_color = "#ffffff" if alpha > 0.5 else "#000000" # White text on dark background, black on light

        html_parts.append(f'<span style="background-color:{bg_color}; color:{text_color}; padding: 2px 4px; margin: 1px; border-radius: 3px; display: inline-block; line-height: 1.4;">{token}</span>')

    return ' '.join(html_parts)

# --- Main App Logic ---
# Load the model and data only once
model, word_to_index, device = load_model_and_data()

st.divider()

# Example options
example_reviews = {
    "Positive Example": "This movie was absolutely amazing! The acting was superb and the storyline kept me engaged throughout. Highly recommended!",
    "Negative Example": "What a terrible waste of time. The plot made no sense and the characters were poorly developed. Awful film.",
    "Mixed Example": "While the special effects were impressive, the dialogue felt weak and the ending was quite disappointing overall.",
    "Subtle Positive": "It wasn't groundbreaking, but I found the characters charming and the story had its moments. A decent watch.",
    "Subtle Negative": "The premise was interesting, but the execution fell flat. It dragged in the middle and never really picked up.",
}

# Example selection
selected_example = st.selectbox(
    "Load an example review or write your own:",
    ["Write your own"] + list(example_reviews.keys()),
    label_visibility="collapsed" # Hide label if context is clear
)

# Set initial text based on selection
initial_text = "" if selected_example == "Write your own" else example_reviews[selected_example]

# User input area
user_review = st.text_area(
    "Enter movie review text here:",
    value=initial_text,
    height=150,
    placeholder="e.g., 'The movie was fantastic, I loved the acting!' or 'A completely boring and predictable plot...'"
)

# Analysis button
if st.button("🔍 Analyze Sentiment", type="primary"):
    if not user_review.strip():
        st.warning("Please enter a review to analyze.")
    elif model is None or word_to_index is None or device is None:
        st.error("Model or data failed to load. Cannot perform analysis. Check error messages above.")
    else:
        with st.spinner("🧠 Analyzing..."):
            try:
                # 1. Preprocess
                processed_review = preprocess_review(user_review, word_to_index)
                processed_review = processed_review.to(device)

                # 2. Predict
                with torch.no_grad():
                    output, attn_weights = model(processed_review)
                    probabilities = F.softmax(output, dim=1)
                    predicted_class = torch.argmax(output, dim=1).item()

                # 3. Interpret Results
                confidence = probabilities[0][predicted_class].item() * 100
                sentiment = "Positive" if predicted_class == 1 else "Negative"
                sentiment_class = "positive" if sentiment == "Positive" else "negative"
                result_box_class = "result-box-positive" if sentiment == "Positive" else "result-box-negative"

                st.divider()
                st.subheader("📊 Analysis Results")

                # Create two columns for the results
                col1, col2 = st.columns([0.6, 0.4]) # Adjust column widths as needed

                with col1:
                    # Display the prediction result in a styled box
                    st.markdown(f"""
                    <div class="result-box {result_box_class}">
                        <h3 class="sentiment-{sentiment_class}" style="margin-top: 0;">Predicted Sentiment: {sentiment}</h3>
                        <p style="font-size: 1.1em; margin-bottom: 0;">Confidence Score: <b>{confidence:.1f}%</b></p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Display highlighted text
                    st.markdown("#### Attention Highlighting")
                    st.markdown(create_highlighted_text(user_review, attn_weights), unsafe_allow_html=True)
                    st.caption("Words are highlighted based on the model's attention. Brighter/darker colors indicate higher attention.")

                with col2:
                    # Display attention visualization (bar chart)
                    st.markdown("#### Attention Weights")
                    if len(user_review.split()) > 0: # Only plot if there are words
                        fig = visualize_attention(user_review, attn_weights)
                        st.pyplot(fig)
                        st.caption("Bar chart showing attention weights assigned to each word (sorted by weight).")
                    else:
                        st.caption("No words to visualize attention for.")

                st.divider()
                # Add interpretation section
                st.subheader("💡 Interpretation")
                tokens = user_review.lower().split()
                if tokens: # Check if there are tokens
                    weights = attn_weights.squeeze().detach().cpu().numpy()[:len(tokens)]
                    word_weights = list(zip(tokens, weights))
                    word_weights.sort(key=lambda x: x[1], reverse=True) # Sort by weight descending

                    # Show top words that influenced the prediction
                    top_n = min(5, len(word_weights))
                    st.markdown(f"The model's **{sentiment.lower()}** prediction appears most influenced by words like:")

                    # Create a readable format for top words
                    top_words_list = [f"'{word}' <span style='color: #6c757d; font-size: small;'>(weight: {weight:.3f})</span>" for word, weight in word_weights[:top_n]]
                    st.markdown("• " + "<br>• ".join(top_words_list), unsafe_allow_html=True)
                else:
                    st.markdown("No words in the input to interpret attention for.")


            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
                st.exception(e) # Show full traceback for debugging

# --- Explanatory Sections ---
st.divider()
with st.expander("ℹ️ How does this model work?"):
    st.markdown("""
    This app uses a **Bidirectional Long Short-Term Memory (LSTM)** neural network combined with an **Attention Mechanism** for sentiment analysis. Here's a simplified breakdown:

    1.  **Word Embeddings**: Each word in the review is converted into a numerical vector (an embedding) that captures its meaning and context relative to other words. Pre-trained embeddings often help the model understand language better from the start.
    2.  **Bidirectional LSTM**: The LSTM network processes the sequence of word embeddings. Being *bidirectional*, it reads the review from start-to-end and end-to-start, allowing it to understand the context of each word based on what comes before *and* after it.
    3.  **Attention Mechanism**: Instead of treating all words equally, the attention mechanism calculates an "importance score" (attention weight) for each word's LSTM output. It learns to focus more on words that are crucial for determining the overall sentiment (e.g., "amazing", "terrible", "not good").
        *   The *Attention Highlighting* and *Attention Weights* visualizations show these scores.
        *   This often improves accuracy, especially for longer reviews, and provides interpretability.
    4.  **Final Classification**: The attention-weighted information is passed through a final layer (a fully connected layer) that outputs the probability of the review being Positive or Negative.

    The combination allows the model to capture long-range dependencies in text and focus on the most relevant parts for sentiment prediction.
    """)

with st.expander("💡 Tips for using the app"):
    st.markdown("""
    *   **Use Clear Sentiment Language**: Words like "great", "awful", "enjoyed", "disappointed" strongly signal sentiment.
    *   **Context Matters**: The model considers surrounding words. "Not good" is different from "good". The BiLSTM helps capture this.
    *   **Check Attention**: If the prediction seems off, look at the highlighted words and the attention weights chart. Does the model focus on the words you'd expect? Sometimes negation ("not", "never") or subtle sarcasm can be challenging.
    *   **File Paths**: If you see errors about missing files, ensure the paths defined at the top of the script (`BASE_DIR` and filenames) correctly point to your `word_to_index.pkl`, `embedding_matrix.pt`, and `lstm_attention_model.pth` files.
    *   **Demo Mode**: If files are missing, the app uses a limited demo vocabulary and random weights, so predictions won't be meaningful.
    """)

# --- Footer ---
st.markdown("---")
st.caption("Sentiment Analysis App | Model: BiLSTM with Attention | Built with Streamlit")


