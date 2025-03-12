import os
import json
import random

import music21 as m21
from sklearn.model_selection import train_test_split

SONG_DATASET_PATH = 'dataset'
MULTIPLE_FILE_DATASET_PATH = 'dataset/processed/raw'
SINGLE_FILE_DATASET_PATH = 'dataset/processed'
MAPPING_PATH = 'dataset/processed'
SEED_FILE_PATH = 'dataset/seeds'

ACCEPT_DUR = [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4]
TIME_STEP = 0.25
SEQUENCE_LENGTH = 64


def load_song(dataset_path):
    songs = []
    # go through all the files and load using m21
    for path, subdir, files in os.walk(dataset_path):
        for file in files:
            if file[-3:] == 'krn':
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)

    return songs


def filter_song(song, acc_dur):
    for note in song.flatten().notesAndRests:
        if note.duration.quarterLength not in acc_dur:
            return False

    return True


def transpose_song(song):
    # get key from song
    part = song.getElementsByClass(m21.stream.Part)
    measures_part_0 = part[0].getElementsByClass(m21.stream.Measure)
    key = measures_part_0[0][4]

    # estimate key if invalid
    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")

    # getting interval for transposition
    if key.mode == 'major':
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == 'minor':
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

    # transpose song by interval
    transposed_song = song.transpose(interval)

    return transposed_song


def encode(song, time_step):
    # p = 60, dur = 1.0 -> [60, "_", "_", "_"]
    encoded_song = []

    for event in song.flatten().notesAndRests:
        # handling notes
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi
        # handling rests
        elif isinstance(event, m21.note.Rest):
            symbol = 'r'

        # converting symbol to time-series notation
        steps = int(event.duration.quarterLength / time_step)
        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append('_')

    encoded_song = ' '.join(map(str, encoded_song))

    return encoded_song


def preprocess(dataset_path, save_dir, train_ratio=0.8, val_ratio=0.06, test_ratio=0.14, r_state=42):
    assert train_ratio + val_ratio + test_ratio == 1.00

    # Loading songs
    print("Loading songs...")
    songs = load_song(dataset_path)
    print(f"Loaded {len(songs)} songs")

    # Creating Train, Validation, Test splits
    songs_train, songs_temp = train_test_split(songs, train_size=train_ratio, random_state=r_state)
    songs_val, songs_test = train_test_split(songs_temp, train_size=val_ratio / (val_ratio + test_ratio), random_state=r_state)
    print("Created splits")

    # Preprocessing Training batch
    for name, split in [('train_set', songs_train), ('val_set', songs_val), ('test_set', songs_test)]:
        for i, song in enumerate(split):
            # Filtering out unacceptable duration
            if not filter_song(song, ACCEPT_DUR):
                continue

            # Transposing songs to C(maj)/A(min)
            song = transpose_song(song)

            # Encode songs in time-series representation
            encoded_song = encode(song, TIME_STEP)

            # Save songs as text files
            save_path = os.path.join(save_dir, name, str(i))
            with open(save_path, 'w') as fp:
                fp.write(encoded_song)

    print("Files saved.")


def load(path):
    with open(path, 'r') as fp:
        song = fp.read()
    return song


def convert_to_single_file(data_path, save_dir, seq_len):
    delim = '/ ' * seq_len
    songs = ''

    # Loading encoded songs and adding delim
    for path, _, files in os.walk(data_path):
        for file in files:
            file_path = os.path.join(path, file)
            song = load(file_path)
            songs = songs + song + ' ' + delim
    songs = songs[:-1]

    # Saving string songs
    with open(save_dir, 'w') as fp:
        fp.write(songs)

    return songs


def create_mapping(songs, map_path):
    mappings = {}

    # Identifying vocab
    songs = songs.split()
    vocab = list(set(songs))

    # Create the mappings
    for i, symbol in enumerate(vocab):
        mappings[symbol] = i

    with open(map_path, 'w') as fp:
        json.dump(mappings, fp, indent=4)


def create_training_data():
    # Create splits and Preprocess the data
    preprocess(SONG_DATASET_PATH, MULTIPLE_FILE_DATASET_PATH, r_state=1)

    # Create a single dataset file for train split
    songs_train = convert_to_single_file(os.path.join(MULTIPLE_FILE_DATASET_PATH, 'train_set'),
                                         os.path.join(SINGLE_FILE_DATASET_PATH, 'train_dataset'),
                                         SEQUENCE_LENGTH)
    create_mapping(songs_train, os.path.join(MAPPING_PATH, 'train_mappings.json'))
    print(len(songs_train))
    print(f'Train Length: {len(songs_train)}')

    # Create a single dataset file for val split
    songs_val = convert_to_single_file(os.path.join(MULTIPLE_FILE_DATASET_PATH, 'val_set'),
                                         os.path.join(SINGLE_FILE_DATASET_PATH, 'val_dataset'),
                                         SEQUENCE_LENGTH)
    create_mapping(songs_val, os.path.join(MAPPING_PATH, 'val_mappings.json'))
    print(len(songs_val))
    print(f'Val Length: {len(songs_val)}')

    # Create a single dataset file for test split
    songs_test = convert_to_single_file(os.path.join(MULTIPLE_FILE_DATASET_PATH, 'test_set'),
                                       os.path.join(SINGLE_FILE_DATASET_PATH, 'test_dataset'),
                                       SEQUENCE_LENGTH)
    create_mapping(songs_test, os.path.join(MAPPING_PATH, 'test_mappings.json'))
    print(f'Test Length: {len(songs_test)}')


def extract_seeds(path, num, output_file=SEED_FILE_PATH):
    try:
        # Get list of all files in directory
        all_files = os.listdir(path)

        # Filter only numeric filenames and convert to integers
        numeric_files = []
        for filename in all_files:
            # Remove extension if it exists and check if it's numeric
            name = os.path.splitext(filename)[0]
            if name.isdigit():
                numeric_files.append(filename)

        # Check if we have enough files
        if len(numeric_files) < num:
            print(f"Warning: Only {len(numeric_files)} files available, less than {num}")
            files_to_process = numeric_files
        else:
            # Randomly select 500 files
            files_to_process = random.sample(numeric_files, num)

        # Process files and store results
        results = []
        for filename in files_to_process:
            file_path = os.path.join(path, filename)
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    # Get first 8 characters (or less if file is shorter)
                    first_eight = content[:20]
                    results.append(first_eight)
            except Exception as e:
                print(f"Error reading file {filename}: {str(e)}")

        # Write results to output file
        with open(output_file, 'a') as out_file:
            for result in results:
                out_file.write(result + '\n')

        print(f"Processed {len(results)} files. Results written to {output_file}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


def create_seeds():
    for file, num in {'train_set': 500, 'val_set': 100, 'test_set': 400}.items():
        path = os.path.join(MULTIPLE_FILE_DATASET_PATH, file)
        extract_seeds(path=path, num=num)


def main():
    # Create the training dataset and mappings
    #create_training_data()

    # Create the seeds file for performance testing
    create_seeds()


if __name__ == '__main__':
    main()