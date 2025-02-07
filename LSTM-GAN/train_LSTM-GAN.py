import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import os
from preprocess import gen_train_seq, SEQUENCE_LENGTH

# Hyperparameters
OUTPUT_UNITS = 45
LATENT_DIM = 100  # Noise vector size for Generator
LSTM_UNITS = 256
LEARNING_RATE = 0.0002
EPOCHS = 50
BATCH_SIZE = 64
SAVE_MODEL_PATH = 'lstm_gan_model.keras'
CHECKPOINT_DIR = 'checkpoints/'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# Generator Model
def build_generator():
    input_noise = layers.Input(shape=(SEQUENCE_LENGTH, LATENT_DIM))
    x = layers.LSTM(LSTM_UNITS, return_sequences=True)(input_noise)
    x = layers.LSTM(LSTM_UNITS)(x)
    x = layers.Dense(OUTPUT_UNITS, activation='softmax')(x)

    model = Model(input_noise, x, name='Generator')
    return model


# Discriminator Model
def build_discriminator():
    input_seq = layers.Input(shape=(SEQUENCE_LENGTH, OUTPUT_UNITS))
    x = layers.LSTM(LSTM_UNITS, return_sequences=True)(input_seq)
    x = layers.LSTM(LSTM_UNITS)(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    model = Model(input_seq, x, name='Discriminator')
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), metrics=['accuracy'])
    return model


# Combine Generator & Discriminator
def build_gan(generator, discriminator):
    discriminator.trainable = False  # Freeze discriminator during GAN training
    noise_input = layers.Input(shape=(SEQUENCE_LENGTH, LATENT_DIM))
    generated_seq = generator(noise_input)
    gan_output = discriminator(generated_seq)

    model = Model(noise_input, gan_output, name='GAN')
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(LEARNING_RATE))
    return model


# Training the GAN with Callbacks
def train_gan():
    # Load training data
    real_inputs, _ = gen_train_seq(SEQUENCE_LENGTH)

    generator = build_generator()
    discriminator = build_discriminator()
    gan = build_gan(generator, discriminator)

    batch_count = real_inputs.shape[0] // BATCH_SIZE

    # Callbacks
    best_generator_path = os.path.join(CHECKPOINT_DIR, 'best_generator.keras')
    best_discriminator_path = os.path.join(CHECKPOINT_DIR, 'best_discriminator.keras')
    early_stopping_patience = 5
    best_g_loss = float('inf')
    best_d_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        for _ in range(batch_count):
            # Generate fake sequences
            noise = np.random.normal(0, 1, (BATCH_SIZE, SEQUENCE_LENGTH, LATENT_DIM))
            fake_sequences = generator.predict(noise)

            # Create labels
            real_labels = np.ones((BATCH_SIZE, 1))
            fake_labels = np.zeros((BATCH_SIZE, 1))

            # Train discriminator
            d_loss_real = discriminator.train_on_batch(real_inputs[:BATCH_SIZE], real_labels)
            d_loss_fake = discriminator.train_on_batch(fake_sequences, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train generator via GAN
            noise = np.random.normal(0, 1, (BATCH_SIZE, SEQUENCE_LENGTH, LATENT_DIM))
            g_loss = gan.train_on_batch(noise, real_labels)

        print(f"Epoch {epoch + 1}/{EPOCHS}, D Loss: {d_loss[0]:.4f}, G Loss: {g_loss:.4f}")

        # Early stopping & checkpointing
        if g_loss < best_g_loss:
            best_g_loss = g_loss
            generator.save(best_generator_path)
            print("Saved best generator model.")
            patience_counter = 0
        else:
            patience_counter += 1

        if d_loss[0] < best_d_loss:
            best_d_loss = d_loss[0]
            discriminator.save(best_discriminator_path)
            print("Saved best discriminator model.")

        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

    generator.save(SAVE_MODEL_PATH)
    print("Training Complete. Model saved.")


if __name__ == '__main__':
    train_gan()