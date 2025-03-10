import json
import tensorflow.keras as keras
import numpy as np

TRAIN_DATASET_PATH = 'processed_data/train_dataset'
VAL_DATASET_PATH = 'processed_data/val_dataset'
TEST_DATASET_PATH = 'processed_data/test_dataset'

TRAIN_MAPPING_PATH = 'processed_data/train_mappings.json'
VAL_MAPPING_PATH = 'processed_data/val_mappings.json'
TEST_MAPPING_PATH = 'processed_data/test_mappings.json'

SEQUENCE_LENGTH = 64

def load(path):
    with open(path, 'r') as fp:
        song = fp.read()
    return song


def song_to_int(songs, map_path):
    int_songs = []

    # Loading mappings
    with open(map_path, 'r') as fp:
        mappings = json.load(fp)

    # Casting songs string to a list
    songs = songs.split()

    # Mapping songs to int
    for symbol in songs:
        int_songs.append(mappings[symbol])

    return int_songs


def gen_train_seq(seq_len, path, map_path):
    # Loading songs and mapping to int
    songs = load(path)
    int_songs = song_to_int(songs, map_path)

    # Generating training sequences
    inputs = []
    targets = []
    num_seq = len(int_songs) - seq_len
    for i in range(num_seq):
        inputs.append(int_songs[i:i+seq_len])
        targets.append(int_songs[i+seq_len])

    # Hot encoding sequences
    vocab_size = len(set(int_songs))
    inputs = keras.utils.to_categorical(inputs, num_classes=vocab_size)
    targets = np.array(targets)

    return inputs, targets


def get_seq(mode='train'):
    if mode=='train':
        return gen_train_seq(SEQUENCE_LENGTH, path=TRAIN_DATASET_PATH, map_path=TRAIN_MAPPING_PATH)
    elif mode=='val':
        return gen_train_seq(SEQUENCE_LENGTH, path=VAL_DATASET_PATH, map_path=VAL_MAPPING_PATH)
    elif mode=='test':
        return gen_train_seq(SEQUENCE_LENGTH, path=TEST_DATASET_PATH, map_path=TEST_MAPPING_PATH)
    else:
        print('Mode Error')

def main():
    inputs, targets = get_seq('test')
    print(inputs.shape)
    print(targets.shape)

if __name__ == '__main__':
    main()