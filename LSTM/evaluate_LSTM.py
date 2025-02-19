import os
import tensorflow.keras as keras
import numpy as np
from dotenv import load_dotenv
from preprocess import gen_train_seq

load_dotenv()

SEQUENCE_LENGTH = int(os.getenv("SEQUENCE_LENGTH"))

MODEL_PATH = 'model_LSTM.h5'

def load_model(model_path):
    """Load the trained model."""
    return keras.models.load_model(model_path)

def evaluate_model(model, inputs, targets):
    """
    Evaluate the model's accuracy on a given dataset.
    :param model: Trained Keras model.
    :param inputs: Input sequences (one-hot encoded).
    :param targets: True next-symbol indices.
    :return: Accuracy of the model.
    """
    # Predict probabilities for each input sequence
    predictions = model.predict(inputs)

    # Get predicted class indices (symbols with the highest probability)
    predicted_indices = np.argmax(predictions, axis=1)

    # Compute accuracy
    accuracy = np.mean(predicted_indices == targets)
    return accuracy

def main():
    # Generate the training sequences (inputs and targets)
    inputs, targets = gen_train_seq(SEQUENCE_LENGTH)

    # Load the trained model
    print("Loading model...")
    model = load_model(MODEL_PATH)

    # Evaluate the model's accuracy
    print("Evaluating model...")
    accuracy = evaluate_model(model, inputs, targets)
    print(f"Model Accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    main()
