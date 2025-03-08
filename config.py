# config.py
EPOCHS = 10
BATCH_SIZE = 32
DATA_PATH = "processed_data/dataset.json"
TRAIN_DATA_PATH = "processed_data/train.json"
VALIDATION_DATA_PATH = "processed_data/validation.json"
TEST_DATA_PATH = "processed_data/test.json"
MODEL_SAVE_PATH = "model/transformer.keras"
MAX_POSITIONS_IN_POSITIONAL_ENCODING = 300

TRANSFORMER_PARAMS = {
    "num_layers": 2,
    "d_model": 64,
    "num_heads": 2,
    "d_feedforward": 128,
    "dropout_rate": 0.1,
}
