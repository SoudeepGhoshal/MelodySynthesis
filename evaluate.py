import os
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
from utils import get_seq
from sklearn.metrics import confusion_matrix
import seaborn as sns

SEQUENCE_LENGTH = 64
MODEL_PATH = 'model/lstm.keras'
EVAL_PLOT_PATH = 'model/plots'

def load_model(model_path):
    """Load the trained model."""
    return keras.models.load_model(model_path)

def evaluate_model(model, inputs, targets):
    """
    Evaluate the model's accuracy on a given dataset and create visualizations.
    :param model: Trained Keras model.
    :param inputs: Input sequences (one-hot encoded).
    :param targets: True next-symbol indices.
    :return: Accuracy of the model, predicted indices
    """
    # Predict probabilities for each input sequence
    predictions = model.predict(inputs)
    
    # Get predicted class indices (symbols with the highest probability)
    predicted_indices = np.argmax(predictions, axis=1)
    
    # Compute accuracy
    accuracy = np.mean(predicted_indices == targets)
    
    return accuracy, predicted_indices

def plot_classification_counts(targets, predictions, accuracy, save_path=os.path.join(EVAL_PLOT_PATH,'classification_counts.png')):
    """Plot bar chart of correct vs incorrect classifications."""
    correct = np.sum(predictions == targets)
    incorrect = len(targets) - correct
    
    plt.figure(figsize=(16, 12))
    plt.bar(['Correct', 'Incorrect'], [correct, incorrect], color=['green', 'red'])
    plt.title(f'Classification Results | Test Accuracy = {accuracy}')
    plt.ylabel('Number of Predictions')
    plt.text(0, correct, str(correct), ha='center', va='bottom')
    plt.text(1, incorrect, str(incorrect), ha='center', va='bottom')
    
    # Create plots directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(targets, predictions, accuracy, save_path=os.path.join(EVAL_PLOT_PATH,'confusion_matrix.png')):
    """Plot confusion matrix of predictions vs actual values."""
    cm = confusion_matrix(targets, predictions)
    
    plt.figure(figsize=(30, 25))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix | Test Accuracy = {accuracy}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Create plots directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def main():
    # Generate the training sequences (inputs and targets)
    inputs_test, targets_test = get_seq(mode='test')

    # Load the trained model
    print("Loading model...")
    model = load_model(MODEL_PATH)

    # Evaluate the model's accuracy
    print("Evaluating model...")
    accuracy, predicted_indices = evaluate_model(model, inputs_test, targets_test)
    print(f"Model Accuracy: {accuracy:.4f}")

    # Generate and save visualizations
    print("Generating visualizations...")
    plot_classification_counts(targets_test, predicted_indices, accuracy)
    plot_confusion_matrix(targets_test, predicted_indices, accuracy)
    print("Visualizations saved in 'plots' directory")

if __name__ == '__main__':
    main()