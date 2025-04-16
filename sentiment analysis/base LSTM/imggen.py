import tensorflow as tf
from tensorflow.keras.utils import plot_model

# Load your saved model
model = tf.keras.models.load_model("sentiment analysis/saved files/sentiment_model.keras")

# Plot the model architecture
plot_model(model, to_file="model_architecture.png", show_shapes=True, show_layer_names=True)

print("Model architecture saved as model_architecture.png")