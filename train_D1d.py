import io
import sys
import json
import psutil
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
#from keras.src.utils import plot_model
from utils import get_seq, SEQUENCE_LENGTH

# Model hyperparameters
OUTPUT_UNITS = 45  # From train_mappings.json
D_MODEL = 256  # Embedding size
NUM_HEADS = 8  # Transformer attention heads
FF_DIM = 512  # Feedforward layer dimension
NUM_LAYERS = 3  # Transformer Encoder layers
LSTM_UNITS = 256  # LSTM Decoder units
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 64

# Paths
ABL = 'D1d'
MODEL_PATH = 'model/' + ABL + '/hybrid_' + ABL + '.keras'
MODEL_ARCH_PATH = 'model/' + ABL + '/hybrid_' + ABL + '_architecture.png'
HISTORY_FILE_PATH = 'model/' + ABL + '/training_history_' + ABL + '.json'


class LearningRateLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        logs = logs or {}
        logs['lr'] = lr


def transformer_encoder(inputs, num_heads, ff_dim):
    """ Transformer Encoder Block """
    x = keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=D_MODEL)(x, x)
    x = keras.layers.Dropout(0.1)(x)
    res = keras.layers.Add()([x, inputs])  # Residual Connection
    x = keras.layers.LayerNormalization(epsilon=1e-6)(res)
    x = keras.layers.Dense(ff_dim, activation='relu')(x)
    x = keras.layers.Dense(D_MODEL)(x)
    x = keras.layers.Dropout(0.1)(x)
    return keras.layers.Add()([x, res])  # Another residual connection


def build_hybrid_model(output_units, d_model, num_heads, ff_dim, num_layers, lstm_units, sequence_length):
    """ Build Transformer Encoder + Enhanced LSTM Decoder Model """

    inputs = keras.layers.Input(shape=(sequence_length, output_units))  # (batch_size, 64, 45)

    # Embedding layer
    x = keras.layers.Dense(d_model)(inputs)

    # Transformer Encoder Stack
    for _ in range(num_layers):
        x = transformer_encoder(x, num_heads, ff_dim)

    # LSTM Decoder
    x = keras.layers.LSTM(lstm_units, return_sequences=True, kernel_regularizer=keras.regularizers.l2(0.0001))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LSTM(lstm_units)(x)
    x = keras.layers.Dropout(0.2)(x)

    # Final output layer
    output = keras.layers.Dense(output_units, activation='softmax')(x)  # (batch_size, 45)

    model = keras.Model(inputs, output)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  metrics=['accuracy'])

    model.summary()

    #plot_model(model, to_file=MODEL_ARCH_PATH, show_shapes=True, show_layer_names=True)

    return model


def log_memory():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"Memory usage: {mem_info.rss / 1024 ** 2:.2f} MB")


def train_model():
    K.clear_session()
    print("Cleared session")
    log_memory()

    print("Loading data...")
    inputs_train, targets_train = get_seq(mode='train')
    inputs_val, targets_val = get_seq(mode='val')
    log_memory()

    if inputs_train is None or targets_train is None:
        print("Error: Training data could not be loaded.")
        return
    if inputs_val is None or targets_val is None:
        print("Error: Validation data could not be loaded.")
        return

    print(f"Train inputs shape: {inputs_train.shape}, Targets shape: {targets_train.shape}")
    print(f"Val inputs shape: {inputs_val.shape}, Targets shape: {targets_val.shape}")
    print("Data loaded.")

    print("Building hybrid model...")
    model = build_hybrid_model(OUTPUT_UNITS, D_MODEL, NUM_HEADS, FF_DIM, NUM_LAYERS, LSTM_UNITS, SEQUENCE_LENGTH)
    print("Model built")
    log_memory()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-5, verbose=1),
        LearningRateLogger()
    ]

    print("Starting training...")
    hist = model.fit(inputs_train, targets_train,
                     epochs=EPOCHS,
                     batch_size=BATCH_SIZE,
                     validation_data=(inputs_val, targets_val),
                     callbacks=callbacks)
    print("Training completed")

    print("Saving model...")
    model.save(MODEL_PATH)
    print("Model saved successfully.")

    print("Saving training history...")
    history_dict = hist.history
    serializable_history = {k: [float(x) for x in v] for k, v in history_dict.items()}
    with open(HISTORY_FILE_PATH, 'w', encoding='utf-8') as f:
        json.dump(serializable_history, f, indent=4)
    print(f"Training history saved to {HISTORY_FILE_PATH}")


if __name__ == '__main__':
    train_model()