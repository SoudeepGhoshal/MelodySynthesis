import io
import sys
import tensorflow.keras as keras
from keras.src.utils import plot_model
from utils import get_seq

OUTPUT_UNITS = 45  # Number of mappings in the mapping.json file
NUM_UNITS = [256]
LOSS = 'sparse_categorical_crossentropy'
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 64

MODEL_PATH = 'model/lstm.keras'
MODEL_ARCH_PATH = 'model/model_architecture.png'
LOG_FILE_PATH = 'model/training_logs.txt'

# Custom callback to save epoch logs to a text file
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


class Tee(object):
    """A class to duplicate output to both terminal and a file."""
    def __init__(self, terminal, file):
        self.terminal = terminal
        self.file = file

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)

    def flush(self):
        self.terminal.flush()
        self.file.flush()


def build_model(out_u, num_u, los, learn_rate):
    # Creating model architecture
    input = keras.layers.Input(shape=(None, out_u))
    x = keras.layers.LSTM(num_u[0], return_sequences=True, kernel_regularizer=keras.regularizers.l2(0.01))(input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LSTM(num_u[0])(x)
    x = keras.layers.Dropout(0.2)(x)

    output = keras.layers.Dense(out_u, activation='softmax')(x)

    model = keras.Model(input, output)

    # Compile model
    model.compile(loss=los,
                  optimizer=keras.optimizers.Adam(learning_rate=learn_rate),
                  metrics=['accuracy', 'categorical_cross_entropy'])

    model.summary()

    # Capture model summary
    model_summary = io.StringIO()
    sys.stdout = model_summary  # Redirect stdout to capture summary
    model.summary()
    sys.stdout = sys.__stdout__  # Reset stdout

    # Save model summary to training logs
    with open(LOG_FILE_PATH, 'w', encoding='utf-8') as f:
        f.write("=== Model Summary ===\n")
        f.write(model_summary.getvalue())
        f.write("=====================\n\n")

    plot_model(
        model,
        to_file=MODEL_ARCH_PATH,
        show_shapes=True,
        show_layer_names=True
    )

    return model

def train_model():
    # Generating training sequences
    inputs_train, targets_train = get_seq(mode='train')
    inputs_val, targets_val = get_seq(mode='val')
    inputs_test, targets_test = get_seq(mode='test')

    # Building the RNN model
    model = build_model(OUTPUT_UNITS, NUM_UNITS, LOSS, LEARNING_RATE)

    # Defining callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-5,verbose=1),
        EpochLogSaver('model/EpochLogSaver.txt')
    ]

    # Open log file in append mode
    with open(LOG_FILE_PATH, 'a', encoding='utf-8') as log_file:
        # Create Tee object to duplicate output
        tee = Tee(sys.stdout, log_file)
        sys.stdout = tee

        # Training the model
        model.fit(inputs_train,
                  targets_train,
                  epochs=EPOCHS,
                  batch_size=BATCH_SIZE,
                  validation_data=(inputs_val, targets_val),
                  callbacks=callbacks)

        # Reset stdout to terminal only
        sys.stdout = sys.__stdout__

    # Save the trained model
    model.save(MODEL_PATH)

if __name__ == '__main__':
    train_model()