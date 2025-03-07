# config.py
EPOCHS = 50
BATCH_SIZE = 64
DATA_PATH = "dataset.json"
MODEL_SAVE_PATH = "model/transformer.keras"
MAX_POSITIONS_IN_POSITIONAL_ENCODING = 300

TRANSFORMER_PARAMS = {
    "num_layers": 2,
    "d_model": 64,
    "num_heads": 2,
    "d_feedforward": 128,
    "dropout_rate": 0.1,
}
