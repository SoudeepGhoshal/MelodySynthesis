import tensorflow.keras as keras
import json
import numpy as np
import music21 as m21
from preprocess import SEQUENCE_LENGTH, MAPPING_PATH

MODEL_PATH = 'lstm_gan_model.keras'  # GAN Generator Model Path

class LSTMGANMelodyGenerator:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

        with open(MAPPING_PATH, 'r') as fp:
            self._mappings = json.load(fp)

        self._start_symbols = ['/'] * SEQUENCE_LENGTH

    def _sample_with_temperature(self, prob, temp):
        pred = np.log(prob + 1e-8) / temp  # Avoid log(0) error
        prob = np.exp(pred) / np.sum(np.exp(pred))
        index = np.random.choice(len(prob), p=prob)
        return index

    def generate_melody(self, seed, num_steps, temperature=1.0):
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed
        seed = [self._mappings[symbol] for symbol in seed]

        for _ in range(num_steps):
            seed = seed[-SEQUENCE_LENGTH:]
            noise = np.random.normal(0, 1, (1, SEQUENCE_LENGTH, 100))  # GAN latent input
            one_hot_seed = keras.utils.to_categorical(seed, num_classes=len(self._mappings))
            one_hot_seed = one_hot_seed[np.newaxis, ...]
            prob = self.model.predict([noise, one_hot_seed])[0]
            output_int = self._sample_with_temperature(prob, temperature)
            seed.append(output_int)
            output_symbol = [k for k, v in self._mappings.items() if v == output_int][0]
            if output_symbol == '/':
                break
            melody.append(output_symbol)
        return melody

    def save_melody(self, melody, step_dur=0.25, file_name='melody_gan.midi'):
        stream = m21.stream.Stream()
        start_symbol = None
        step_counter = 1

        for i, symbol in enumerate(melody):
            if symbol != '_' or i + 1 == len(melody):
                if start_symbol is not None:
                    duration = step_dur * step_counter
                    if start_symbol == 'r':
                        event = m21.note.Rest(quarterLength=duration)
                    else:
                        event = m21.note.Note(int(start_symbol), quarterLength=duration)
                    stream.append(event)
                    step_counter = 1
                start_symbol = symbol
            else:
                step_counter += 1
        stream.write('midi', file_name)

if __name__ == '__main__':
    generator = LSTMGANMelodyGenerator()
    seed = '55 _ 55 _ 60 _ 60 _ _ _ 60 _'
    melody = generator.generate_melody(seed, 1500, temperature=1.0)
    print(melody)
    generator.save_melody(melody)
    print("Melody saved...")