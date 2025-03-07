import tensorflow.keras as keras
from keras.src.utils import plot_model
from preprocess import gen_train_seq, SEQUENCE_LENGTH

OUTPUT_UNITS = 45  # Number of mappings in the mapping.json file
NUM_UNITS = [256]
LOSS = 'sparse_categorical_crossentropy'
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 64
SAVE_MODEL_PATH = 'models/model_LSTM.keras'
LOG_FILE_PATH = 'training_logs.txt'  # Path to save the training logs

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

def build_model(out_u, num_u, los, learn_rate):
    # Creating model architecture
    input = keras.layers.Input(shape=(None, out_u))
    x = keras.layers.LSTM(num_u[0], return_sequences=True, kernel_regularizer=keras.regularizers.l2(0.01))
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

    plot_model(
        model,
        to_file='model_architecture.png',
        show_shapes=True,
        show_layer_names=True
    )

    return model

def train_model():
    # Generating training sequences
    inputs_train, target_train, inputs_val, target_val, inputs_test, target_test = gen_train_seq(seq_len=SEQUENCE_LENGTH)

    # Building the RNN model
    model = build_model(OUTPUT_UNITS, NUM_UNITS, LOSS, LEARNING_RATE)

    # Defining callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(SAVE_MODEL_PATH, save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-5,verbose=1),
        EpochLogSaver(LOG_FILE_PATH)
    ]

    # Training the model
    model.fit(inputs_train,
              target_train,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              validation_data=(inputs_val, target_val),
              callbacks=callbacks)

    # Save the trained model
    model.save(SAVE_MODEL_PATH)

if __name__ == '__main__':
    train_model()