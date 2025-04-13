import os
import tensorflow.keras as keras
import json
import numpy as np
import music21 as m21

SEQUENCE_LENGTH = 64
MAPPING_PATH = 'processed_data/train_mappings.json'
SEEDS_PATH = 'melodies/seeds'
OUTPUTS_PATH = 'melodies/outputs'
MODEL_PATH = 'model/transformer.keras'

class MelodyGenerator:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

        with open(MAPPING_PATH, 'r') as fp:
            self._mappings = json.load(fp)

        self._start_symbols = ['/'] * SEQUENCE_LENGTH

    def _sample_with_temperature(self, prob, temp):
        # Temperature = infinity: Remodel prob distribution as homogeneous distribution
        # Temperature = 0: (deterministic) Remodel prob distribution to make prob of symbol with the highest prob in original distribution as 1
        # Temperature = 1: No change

        pred = np.log(prob) / temp
        prob = np.exp(pred) / np.sum(np.exp(pred))

        choices = range(len(prob))
        index = np.random.choice(choices, p=prob)

        return index

    def gen_mel(self, seed, num_steps, max_seq_length, temperature):
        # Creating a seed with start symbol
        seed = seed.split(' ')
        # Handle case where seed end on single digit to avoid KeyError
        if seed and seed[-1].isdigit() and len(seed[-1]) == 1:
            seed.pop()
        print(f'Generating with seed: {seed}')

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


def convert_seeds_to_outputs():
    mg = MelodyGenerator()

    try:
        with open(SEEDS_PATH, "r") as file:
            seeds = [line.strip() for line in file.readlines()]
        print(seeds)
    except FileNotFoundError:
        print(f"Error: The file at {SEEDS_PATH} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    results = []
    counter = 0
    with open(OUTPUTS_PATH, 'a') as out_file:
        for seed in seeds[991:]:
            gen = mg.gen_mel(seed, 1500, SEQUENCE_LENGTH, 0.70)
            gen_string = " ".join(gen)
            print(f'Output generated: {gen_string}')
            counter += 1

            out_file.write(gen_string + '\n')
            print(f'Output ({counter}/{len(seeds)}) saved.')

            results.append(gen_string)
    print(f'All outputs saved to {OUTPUTS_PATH}')

    return results

def gen_melody_from_seed():
    mg = MelodyGenerator()
    seed = input('Enter seed: ').strip() # Example: '64 _ _ _ 72 _ _ _ 72'
    print(seed)
    melody = mg.gen_mel(seed, 1500, SEQUENCE_LENGTH, 0.5)
    print(melody)
    mg.save_mel(melody)
    print("Melody saved...")


if __name__ == '__main__':
    outputs = convert_seeds_to_outputs()
    print(outputs)

    #gen_melody_from_seed()