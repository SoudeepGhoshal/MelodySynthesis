import os
import tensorflow.keras as keras
import json
import numpy as np
import music21 as m21
from dotenv import load_dotenv

load_dotenv()

SEQUENCE_LENGTH = int(os.getenv("SEQUENCE_LENGTH"))
MAPPING_PATH = os.getenv('MAPPING_PATH')

MODEL_PATH = 'LSTM/model_GRU.h5'

class MelodyGenerator:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

        with open(MAPPING_PATH, 'r') as fp:
            self._mappings = json.load(fp)

        self._start_symbols = ['/'] * SEQUENCE_LENGTH

    def _sample_with_temperature(self, prob, temp):
        # Temperature = infinity: Remodel prob distribution as homogeneous distribution
        # Temperature = 0: (deterministic) Remodel prob distribution to make prob of symbol with highest prob in original distribution as 1
        # Temperature = 1: No change

        pred = np.log(prob) / temp
        prob = np.exp(pred) / np.sum(np.exp(pred))

        choices = range(len(prob))
        index = np.random.choice(choices, p=prob)

        return index

    def gen_mel(self, seed, num_steps, max_seq_length, temperature):
        # Creating a seed with start symbol
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed

        # Mapping seeds to integers
        seed = [self._mappings[symbol] for symbol in seed]

        for _ in range(num_steps):
            # Limiting the seed to max_seq_length
            seed = seed[-max_seq_length:]

            # One-hot encoding the seeds
            one_hot_seed = keras.utils.to_categorical(seed, num_classes=len(self._mappings))
            #(max_seq_len, no of symbols)
            one_hot_seed = one_hot_seed[np.newaxis, ...]
            #(1, max_seq_len, no of symbols)

            # Making a prediction
            prob = self.model.predict(one_hot_seed)[0]
            #(0.1, 0.2, 0.1, 0.6) --adding up gives--> 1
            #Taking the index with the highest probability for the next prediction would be very rigid, hence we use temperature for flexibility
            output_int = self._sample_with_temperature(prob, temperature)

            # Update seed
            seed.append(output_int)

            # Mapping integer to out encoding
            output_symbol = [k for k, v in self._mappings.items() if v == output_int][0]

            # Checking whether we're at the end of a melody
            if output_symbol == '/':
                break;

            # Updating melody
            melody.append(output_symbol)

        return melody

    def save_mel(self, melody, step_dur=0.25, format='midi', file_name='melody.midi'):
        # Creating a m21 string
        stream = m21.stream.Stream()

        # Passing all the symbols in the melody and create note/rest objects
        start_symbol = None
        step_counter = 1

        for i, symbol in enumerate(melody):
            # Handling case having a note/rest
            if symbol != '_' or i + 1 == len(melody):
                # Ensuring we're handling note/rest beyond the first one
                if start_symbol is not None:
                    quarter_len_dur = step_dur * step_counter

                    # Handling rest
                    if start_symbol == 'r':
                        m21_event = m21.note.Rest(quarterLength=quarter_len_dur)

                    # Handling note
                    else:
                        m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_len_dur)

                    stream.append(m21_event)

                    # Resetting the step counter
                    step_counter = 1

                start_symbol = symbol

            # Handling case having prolongation
            else:
                step_counter += 1

        # Writing the m21 string to a midi file
        stream.write(format, file_name)

if __name__ == '__main__':
    mg = MelodyGenerator()
    seed = '55 _ 55 _ 60 _ 60 _ _ _ 60 _'
    melody = mg.gen_mel(seed, 1500, SEQUENCE_LENGTH, 1)
    print(melody)
    mg.save_mel(melody)
    print("Melody saved...")