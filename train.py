"""
This file contains the training pipeline for a Transformer model specialized in
melody generation. It includes functions to calculate loss, perform training steps,
and orchestrate the training process over multiple epochs. The script also
demonstrates the use of the MelodyGenerator class to generate a melody after training.

The training process uses a custom implementation of the Transformer model,
defined in the 'transformer.py' module, and prepares data using the
MelodyPreprocessor class from 'melodypreprocessor.py'.

Global parameters such as the number of epochs, batch size, and path to the dataset
are defined. The script supports dynamic padding of sequences and employs the
Sparse Categorical Crossentropy loss function for model training.

For simplicity's sake training does not deal with masking of padded values
in the encoder and decoder. Also, look-ahead masking is not implemented.
Both of these are left as an exercise for the student.

Key Functions:
- _calculate_loss_function: Computes the loss between actual and predicted sequences.
- _train_step: Executes a single training step, including forward pass and backpropagation.
- train: Runs the training loop over the entire dataset for a given number of epochs.
- _right_pad_sequence_once: Utility function for padding sequences.

The script concludes by instantiating the Transformer model, conducting the training,
and generating a sample melody using the trained model.
"""

import tensorflow as tf
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
import config
from melody_preprocessor import MelodyPreprocessor
from transformer import Transformer
import json

# Global parameters
EPOCHS = config.EPOCHS
BATCH_SIZE = config.BATCH_SIZE
DATA_PATH = config.DATA_PATH
MODEL_PATH = config.MODEL_SAVE_PATH 
MAX_POSITIONS_IN_POSITIONAL_ENCODING = config.MAX_POSITIONS_IN_POSITIONAL_ENCODING

# Data preparation
# Instantiate preprocessors
train_preprocessor = MelodyPreprocessor(config.TRAIN_DATA_PATH, batch_size=BATCH_SIZE)
val_preprocessor = MelodyPreprocessor(config.VALIDATION_DATA_PATH, batch_size=BATCH_SIZE)
test_preprocessor = MelodyPreprocessor(config.TEST_DATA_PATH, batch_size=BATCH_SIZE)
# Fit tokenizer on training data only
train_melodies_raw = train_preprocessor._load_dataset()
parsed_train_melodies = [train_preprocessor._parse_melody(melody) for melody in train_melodies_raw]
train_preprocessor.fit_tokenizer()

# Reuse tokenizer for val and test preprocessors
val_preprocessor.tokenizer = train_preprocessor.tokenizer
test_preprocessor.tokenizer = train_preprocessor.tokenizer

# Create datasets using the shared tokenizer
train_dataset = train_preprocessor.create_training_dataset()
val_dataset = val_preprocessor.create_training_dataset()
test_dataset = test_preprocessor.create_training_dataset()
vocab_size = train_preprocessor.number_of_tokens_with_padding
print("Vocab size:", vocab_size)
print("Validation vocab size:", val_preprocessor.number_of_tokens_with_padding)
print("Test vocab size:", test_preprocessor.number_of_tokens_with_padding)


# Instantiate Transformer model
transformer_model = Transformer(
    num_layers=2,
    d_model=64,
    num_heads=2,
    d_feedforward=128,
    input_vocab_size=vocab_size,
    target_vocab_size=vocab_size,
    max_num_positions_in_pe_encoder=MAX_POSITIONS_IN_POSITIONAL_ENCODING,
    max_num_positions_in_pe_decoder=MAX_POSITIONS_IN_POSITIONAL_ENCODING,
    dropout_rate=0.1,
)

optimizer = Adam()
loss_function = SparseCategoricalCrossentropy(from_logits=True)

@tf.function
def train_step(input_seq, target_seq):
    target_input = target_seq[:, :-1]
    target_real = target_seq[:, 1:]

    with tf.GradientTape() as tape:
        predictions = transformer_model(input_seq, target_input, True, None, None, None)
        loss = loss_function(target_real, predictions)

    gradients = tape.gradient(loss, transformer_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer_model.trainable_variables))
    return loss

def evaluate(dataset):
    total_loss = 0
    total_batches = 0
    for input_seq, target_seq in dataset:
        target_input = target_seq[:, :-1]
        target_real = target_seq[:, 1:]
        predictions = transformer_model(input_seq, target_input, False, None, None, None)
        loss = loss_function(target_real, predictions)
        total_loss += loss.numpy()
        total_batches += 1
    return total_loss / total_batches

# Training loop with validation evaluation after each epoch
for epoch in range(EPOCHS):
    total_loss, batches_counted = 0, 0
    for batch, (input_seq, target_seq) in enumerate(train_preprocessor.create_training_dataset()):
        batch_loss = train_step(input_seq, target_seq)
        total_loss += batch_loss.numpy()

    avg_train_loss = total_loss / (batch + 1)

    # Evaluate on validation set after each epoch
    val_loss = evaluate(val_preprocessor.create_training_dataset())

    print(f"Epoch {epoch+1}: Train Loss {avg_train_loss:.4f}, Validation Loss {val_loss:.4f}")

transformer_model.save_weights(config.MODEL_SAVE_PATH)

# Final evaluation on test set after training completes:
final_test_loss = evaluate(test_preprocessor.create_training_dataset())
print(f"Final Test Loss: {final_test_loss:.4f}")
