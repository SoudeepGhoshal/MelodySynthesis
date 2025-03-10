from melody_preprocessor import MelodyPreprocessor
from transformer import Transformer
import config
import tensorflow as tf

class MelodyGenerator:
    """
    Class to generate melodies using a trained Transformer model.

    This class encapsulates the inference logic for generating melodies
    based on a starting sequence.
    """

    def __init__(self, transformer, tokenizer, max_length=64):
        """
        Initializes the MelodyGenerator.

        Parameters:
            transformer (Transformer): The trained Transformer model.
            tokenizer (Tokenizer): Tokenizer used for encoding melodies.
            max_length (int): Maximum length of the generated melodies.
        """
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.max_length = max_length

    def generate(self, start_sequence):
        """
        Generates a melody based on a starting sequence.

        Parameters:
            start_sequence (list of str): The starting sequence of the melody.

        Returns:
            str: The generated melody.
        """
        input_tensor = self._get_input_tensor(start_sequence)

        num_notes_to_generate = self.max_length - len(input_tensor[0])

        for _ in range(num_notes_to_generate):
            predictions = self.transformer(
                input_tensor, input_tensor, False, None, None, None
            )
            predicted_note = self._get_note_with_highest_score(predictions)
            input_tensor = self._append_predicted_note(
                input_tensor, predicted_note
            )

        generated_melody = self._decode_generated_sequence(input_tensor)

        return generated_melody

    def _get_input_tensor(self, start_sequence):
        """
        Gets the input tensor for the Transformer model.

        Parameters:
            start_sequence (list of str): The starting sequence of the melody.

        Returns:
            input_tensor (tf.Tensor): The input tensor for the model.
        """
        input_sequence = self.tokenizer.texts_to_sequences([start_sequence])
        input_tensor = tf.convert_to_tensor(input_sequence, dtype=tf.int64)
        return input_tensor

    def _get_note_with_highest_score(self, predictions, temperature=0.7):
        """
        Gets the note with the highest score from the predictions with temperature sampling.
        """
        latest_predictions = predictions[:, -1, :]
        scaled_predictions = latest_predictions / temperature
        probabilities = tf.nn.softmax(scaled_predictions)
        predicted_note_index = tf.random.categorical(probabilities, num_samples=1)
        return predicted_note_index.numpy()[0][0]

    def _append_predicted_note(self, input_tensor, predicted_note):
        """
        Appends the predicted note to the input tensor.

        Parameters:
            input_tensor (tf.Tensor): The input tensor for the model.

        Returns:
            (tf.Tensor): The input tensor with the predicted note
        """
        return tf.concat([input_tensor, [[predicted_note]]], axis=-1)

    def _decode_generated_sequence(self, generated_sequence):
        """
        Decodes the generated sequence of notes.

        Parameters:
            generated_sequence (tf.Tensor): Tensor with note indexes generated.

        Returns:
            generated_melody (str): The decoded sequence of notes.
        """
        generated_sequence_array = generated_sequence.numpy()
        generated_melody = self.tokenizer.sequences_to_texts(
            generated_sequence_array
        )[0]
        return generated_melody

# Load and parse training melodies explicitly to fit tokenizer (same as training)
train_preprocessor = MelodyPreprocessor(config.TRAIN_DATA_PATH)
train_raw_melodies = train_preprocessor._load_dataset()
parsed_train_melodies = [train_preprocessor._parse_melody(melody) for melody in train_raw_melodies]

# Fit tokenizer ONLY on training melodies (important!)
train_preprocessor.tokenizer.fit_on_texts(parsed_train_melodies)

train_preprocessor._set_number_of_tokens()
tokenized_train_melodies = train_preprocessor._tokenize_and_encode_melodies(parsed_train_melodies)
train_preprocessor._set_max_melody_length(tokenized_train_melodies)

# Get vocab size from preprocessor after tokenizer fitting
vocab_size = train_preprocessor.number_of_tokens_with_padding

# Instantiate Transformer model (same parameters as training)
transformer_model = Transformer(
    num_layers=config.TRANSFORMER_PARAMS["num_layers"],
    d_model=config.TRANSFORMER_PARAMS["d_model"],
    num_heads=config.TRANSFORMER_PARAMS["num_heads"],
    d_feedforward=config.TRANSFORMER_PARAMS["d_feedforward"],
    input_vocab_size=vocab_size,
    target_vocab_size=vocab_size,
    max_num_positions_in_pe_encoder=config.MAX_POSITIONS_IN_POSITIONAL_ENCODING,
    max_num_positions_in_pe_decoder=config.MAX_POSITIONS_IN_POSITIONAL_ENCODING,
    dropout_rate=config.TRANSFORMER_PARAMS["dropout_rate"],
)

# Build the model by calling it once with dummy data (this initializes variables)
dummy_input = tf.zeros((1, 10), dtype=tf.int64)  # adjust shape if needed
transformer_model(dummy_input, dummy_input, False, None, None, None)

# Now load weights safely after building the model
transformer_model.load_weights(config.MODEL_SAVE_PATH)

# Generate new melody  
melody_generator = MelodyGenerator(transformer_model, train_preprocessor.tokenizer)

start_sequence = ["C4-0.5", "B3-0.5", "G3-0.5", "C4-0.5"]
generated_melody = melody_generator.generate(start_sequence=start_sequence)

print("Generated melody:", generated_melody)

# Save generated melody to JSON  
import json  
with open('outputs/output.json', 'w') as f:  
    json.dump([generated_melody.replace(" ", ", ")], f, indent=2)  

print("Melody saved to output.json")
