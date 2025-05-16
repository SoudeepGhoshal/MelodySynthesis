import io
import sys
import json
import tensorflow as tf
import tensorflow.keras as keras
from keras.src.utils import plot_model
import numpy as np
import psutil
import tensorflow.keras.backend as K
from utils import get_seq, SEQUENCE_LENGTH

OUTPUT_UNITS = 45  # Vocabulary size from train_mappings.json
D_MODEL = 256  # Embedding size
NUM_HEADS = 8  # Attention heads
FF_DIM = 1024  # Feedforward layer dimension
NUM_LAYERS = 4  # Decoder layers
LEARNING_RATE = 0.001  # Base learning rate
EPOCHS = 50
BATCH_SIZE = 64
WARMUP_STEPS = 4000

MODEL_PATH = 'model/transformer.keras'
MODEL_ARCH_PATH = 'model/transformer_architecture.png'
LOG_FILE_PATH = 'model/training_logs.txt'
HISTORY_FILE_PATH = 'model/training_history.json'


class LearningRateLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        logs = logs or {}
        logs['lr'] = lr

@tf.keras.utils.register_keras_serializable()
class CustomLearningRateSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps):
        super(CustomLearningRateSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {
            'd_model': int(self.d_model),
            'warmup_steps': self.warmup_steps
        }

def positional_encoding(sequence_length, d_model):
    position = tf.range(sequence_length, dtype=tf.float32)[:, tf.newaxis]
    i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
    angle_rads = position / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
    sines = tf.math.sin(angle_rads[:, 0::2])
    cosines = tf.math.cos(angle_rads[:, 1::2])
    pos_encoding = tf.concat([sines, cosines], axis=-1)
    return pos_encoding[tf.newaxis, :, :]

def transformer_decoder(inputs, num_heads, ff_dim, sequence_length):
    pos_encoding = positional_encoding(sequence_length, D_MODEL)
    x = inputs + pos_encoding
    x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=D_MODEL)(x, x)
    x = keras.layers.Dropout(0.2)(x)
    res = x + inputs
    x = keras.layers.LayerNormalization(epsilon=1e-6)(res)
    x = keras.layers.Dense(ff_dim, activation='relu')(x)
    x = keras.layers.Dense(D_MODEL)(x)
    x = keras.layers.Dropout(0.2)(x)
    return x + res

def build_model(output_units, d_model, num_heads, ff_dim, num_layers, sequence_length):
    inputs = keras.layers.Input(shape=(sequence_length, output_units))  # (batch_size, 64, 45)
    x = keras.layers.Dense(d_model)(inputs)
    for _ in range(num_layers):
        x = transformer_decoder(x, num_heads, ff_dim, sequence_length)
    x = x[:, -1, :]  # Predict next token (batch_size, D_MODEL)
    output = keras.layers.Dense(output_units, activation='softmax', dtype='float32')(x)  # (batch_size, 45)

    model = keras.Model(inputs, output)
    optimizer = keras.optimizers.Adam(learning_rate=CustomLearningRateSchedule(d_model, WARMUP_STEPS))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'],
                  jit_compile=True)

    model.summary()

    plot_model(model, to_file=MODEL_ARCH_PATH, show_shapes=True, show_layer_names=True)

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
    inputs_test, targets_test = get_seq(mode='test')
    log_memory()

    if inputs_train is None or targets_train is None:
        print("Error: Training data could not be loaded.")
        return
    if inputs_val is None or targets_val is None:
        print("Error: Validation data could not be loaded.")
        return

    print(f"Train inputs shape: {inputs_train.shape}, Targets shape: {targets_train.shape}")
    print(f"Val inputs shape: {inputs_val.shape}, Targets shape: {targets_val.shape}")
    print("Created datasets")
    log_memory()

    print("Building model...")
    model = build_model(OUTPUT_UNITS, D_MODEL, NUM_HEADS, FF_DIM, NUM_LAYERS, SEQUENCE_LENGTH)
    print("Model built")
    log_memory()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True, verbose=1),
        LearningRateLogger()
    ]

    print("Starting training...")
    hist = model.fit(inputs_train, targets_train,
                     epochs=EPOCHS,
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