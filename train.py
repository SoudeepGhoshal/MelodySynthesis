import tensorflow.keras as keras
from preprocess import gen_train_seq, SEQUENCE_LENGTH

OUTPUT_UNITS = 45 # Number of mappings in the mapping.json file
NUM_UNITS = [256]
LOSS = 'sparse_categorical_crossentropy'
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 64
SAVE_MODEL_PATH = 'model.h5'

def build_model(out_u, num_u, los, learn_rate):
    # Creating model architecture
    input = keras.layers.Input(shape=(None, out_u))
    x = keras.layers.GRU(num_u[0])(input)
    x = keras.layers.Dropout(0.2)(x)

    output = keras.layers.Dense(out_u, activation='softmax')(x)

    model = keras.Model(input, output)

    # Compile model
    model.compile(loss=los,
                  optimizer=keras.optimizers.Adam(learning_rate=learn_rate),
                  metrics=["accuracy"])

    model.summary()

    return model

def train_model():
    # Generating training sequences
    inputs, targets = gen_train_seq(SEQUENCE_LENGTH)

    # Building the RNN model
    model = build_model(OUTPUT_UNITS, NUM_UNITS, LOSS, LEARNING_RATE)

    # Training the model
    model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # Save the trained model
    model.save(SAVE_MODEL_PATH)

if __name__ == '__main__':
    train_model()