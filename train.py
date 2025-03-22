import io
import sys
import tensorflow as tf
import tensorflow.keras as keras
from keras.src.utils import plot_model
from utils import get_seq, SEQUENCE_LENGTH

OUTPUT_UNITS = 45  # Vocabulary size from train_mappings.json
D_MODEL = 256  # Embedding size
NUM_HEADS = 8  # Attention heads
FF_DIM = 512  # Feedforward layer dimension
NUM_LAYERS = 3  # Decoder layers
LEARNING_RATE = 0.0001  # Adjusted for stability
EPOCHS = 50
BATCH_SIZE = 32  # Adjusted for efficiency

MODEL_PATH = 'model/transformer.keras'
MODEL_ARCH_PATH = 'model/transformer_architecture.png'
LOG_FILE_PATH = 'model/training_logs.txt'

class EpochLogSaver(keras.callbacks.Callback):
    def __init__(self, log_file):
        super(EpochLogSaver, self).__init__()
        self.log_file = log_file

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        with open(self.log_file, 'a') as f:
            f.write(f"Epoch {epoch + 1}\n")
            f.write(f" - loss: {logs.get('loss'):.4f}\n")
            f.write(f" - accuracy: {logs.get('accuracy'):.4f}\n")
            f.write(f" - val_loss: {logs.get('val_loss'):.4f}\n")
            f.write(f" - val_accuracy: {logs.get('val_accuracy'):.4f}\n")
            f.write("\n")

def transformer_decoder(inputs, num_heads, ff_dim):
    x = keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=D_MODEL)(x, x)
    x = keras.layers.Dropout(0.1)(x)
    res = x + inputs
    x = keras.layers.LayerNormalization(epsilon=1e-6)(res)
    x = keras.layers.Dense(ff_dim, activation='relu')(x)
    x = keras.layers.Dense(D_MODEL)(x)
    x = keras.layers.Dropout(0.1)(x)
    return x + res

def build_model(output_units, d_model, num_heads, ff_dim, num_layers, sequence_length):
    inputs = keras.layers.Input(shape=(sequence_length, output_units))  # (batch_size, 64, 45)

    # Transform to embedding size
    x = keras.layers.Dense(d_model)(inputs)

    # Decoder stack with 3 layers
    for _ in range(num_layers):
        x = transformer_decoder(x, num_heads, ff_dim)

    # Predict next token from the last timestep
    x = x[:, -1, :]  # (batch_size, D_MODEL)
    output = keras.layers.Dense(output_units, activation='softmax')(x)  # (batch_size, 45)

    model = keras.Model(inputs, output)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  metrics=['accuracy'])

    model.summary()

    # Capture model summary
    model_summary = io.StringIO()
    sys.stdout = model_summary
    model.summary()
    sys.stdout = sys.__stdout__

    with open(LOG_FILE_PATH, 'w', encoding='utf-8') as f:
        f.write("=== Model Summary ===\n")
        f.write(model_summary.getvalue())
        f.write("=====================\n\n")

    plot_model(model, to_file=MODEL_ARCH_PATH, show_shapes=True, show_layer_names=True)
    return model

def train_model():
    inputs_train, targets_train = get_seq(mode='train')
    inputs_val, targets_val = get_seq(mode='val')
    inputs_test, targets_test = get_seq(mode='test')

    if inputs_train is None or targets_train is None:
        print("Error: Training data could not be loaded.")
        return
    if inputs_val is None or targets_val is None:
        print("Error: Validation data could not be loaded.")
        return

    print(f"Train inputs shape: {inputs_train.shape}, Targets shape: {targets_train.shape}")
    print(f"Val inputs shape: {inputs_val.shape}, Targets shape: {targets_val.shape}")

    model = build_model(OUTPUT_UNITS, D_MODEL, NUM_HEADS, FF_DIM, NUM_LAYERS, SEQUENCE_LENGTH)

    # Learning rate warmup
    lr_schedule = keras.callbacks.LearningRateScheduler(
        lambda epoch: min(LEARNING_RATE * (epoch + 1) / 5, LEARNING_RATE)  # Warmup for 5 epochs
    )

    model = keras.Model(inputs, output)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  metrics=['accuracy'])

    model.summary()

    # Capture model summary
    model_summary = io.StringIO()
    sys.stdout = model_summary
    model.summary()
    sys.stdout = sys.__stdout__

    with open(LOG_FILE_PATH, 'w', encoding='utf-8') as f:
        f.write("=== Model Summary ===\n")
        f.write(model_summary.getvalue())
        f.write("=====================\n\n")

    plot_model(model, to_file=MODEL_ARCH_PATH, show_shapes=True, show_layer_names=True)
    return model

def train_model():
    inputs_train, targets_train = get_seq(mode='train')
    inputs_val, targets_val = get_seq(mode='val')
    inputs_test, targets_test = get_seq(mode='test')

    if inputs_train is None or targets_train is None:
        print("Error: Training data could not be loaded.")
        return
    if inputs_val is None or targets_val is None:
        print("Error: Validation data could not be loaded.")
        return

    print(f"Train inputs shape: {inputs_train.shape}, Targets shape: {targets_train.shape}")
    print(f"Val inputs shape: {inputs_val.shape}, Targets shape: {targets_val.shape}")

    model = build_model(OUTPUT_UNITS, D_MODEL, NUM_HEADS, FF_DIM, NUM_LAYERS, SEQUENCE_LENGTH)

    # Learning rate warmup
    lr_schedule = keras.callbacks.LearningRateScheduler(
        lambda epoch: min(LEARNING_RATE * (epoch + 1) / 5, LEARNING_RATE)  # Warmup for 5 epochs
    )

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-5, verbose=1),
        EpochLogSaver(LOG_FILE_PATH),
        lr_schedule
    ]

    model.fit(inputs_train, targets_train,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              validation_data=(inputs_val, targets_val),
              callbacks=callbacks)

    model.save(MODEL_PATH)

if __name__ == '__main__':
    train_model()