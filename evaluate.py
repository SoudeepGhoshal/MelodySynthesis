import os
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
from utils import get_seq
from sklearn.metrics import confusion_matrix
import seaborn as sns
import json

SEQUENCE_LENGTH = 64
MODEL_PATH = 'model/transformer.keras'
EVAL_PLOT_PATH = 'model/plots'
TRAINING_HISTORY_PATH = 'model/training_history.json'


def load_model(model_path):
    """Load the trained model."""
    return keras.models.load_model(model_path)


def evaluate_model(model, inputs, targets):
    """Evaluate the model's accuracy on a given dataset."""
    predictions = model.predict(inputs)
    predicted_indices = np.argmax(predictions, axis=1)
    accuracy = np.mean(predicted_indices == targets)
    return accuracy, predicted_indices


def plot_classification_counts(targets, predictions, accuracy,
                               save_path=os.path.join(EVAL_PLOT_PATH, 'classification_counts.png')):
    """Plot bar chart of correct vs incorrect classifications."""
    correct = np.sum(predictions == targets)
    incorrect = len(targets) - correct
    plt.figure(figsize=(16, 12))
    plt.bar(['Correct', 'Incorrect'], [correct, incorrect], color=['green', 'red'])
    plt.title(f'Classification Results | Test Accuracy = {accuracy:.4f}')
    plt.ylabel('Number of Predictions')
    plt.text(0, correct, str(correct), ha='center', va='bottom')
    plt.text(1, incorrect, str(incorrect), ha='center', va='bottom')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(targets, predictions, accuracy,
                          save_path=os.path.join(EVAL_PLOT_PATH, 'confusion_matrix.png')):
    """Plot confusion matrix of predictions vs actual values."""
    cm = confusion_matrix(targets, predictions)
    plt.figure(figsize=(30, 25))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix | Test Accuracy = {accuracy:.4f}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def load_training_history(json_path):
    """Load training history from a JSON file."""
    if not os.path.exists(json_path):
        print(f"Training history file not found at {json_path}")
        return None

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            history = json.load(f)

        # Verify that required keys are present
        required_keys = {'accuracy', 'loss', 'val_accuracy', 'val_loss'}
        if not all(key in history for key in required_keys):
            print(f"Training history JSON is missing required keys. Found: {list(history.keys())}")
            return None

        return history
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON file: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error loading training history: {e}")
        return None


def plot_training_history(history, save_path=os.path.join(EVAL_PLOT_PATH, 'training_history.png')):
    """Plot training and validation accuracy/loss curves from history."""
    if not history:
        return

    epochs = range(1, len(history['accuracy']) + 1)

    plt.figure(figsize=(12, 8))

    # Plot accuracy
    plt.subplot(2, 1, 1)
    plt.plot(epochs, history['accuracy'], label='Training Accuracy')
    plt.plot(epochs, history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot loss
    plt.subplot(2, 1, 2)
    plt.plot(epochs, history['loss'], label='Training Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def main():
    # Generate the test sequences
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

    # Load training history and plot
    history = load_training_history(TRAINING_HISTORY_PATH)
    if history:
        plot_training_history(history)
        print("Training history plot generated")
    else:
        print("Could not generate training history plot: No valid history data")

    print("Visualizations saved in 'plots' directory")


if __name__ == '__main__':
    main()