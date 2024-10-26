from pathlib import Path


class Config:
    # Model configuration
    INPUT_SIZE = 512
    INPUT_CHANNELS = 3
    DUAL_INPUT_CHANNELS = 6  # For model2 which takes concatenated input
    NUM_CLASSES = 3

    # File paths
    BASE_DIR = Path(__file__).parent
    MODEL_DIR = BASE_DIR / "weights"
    GCNET_WEIGHTS = MODEL_DIR / "gcnet.pt"
    DRNET_WEIGHTS = MODEL_DIR / "drnet.pt"

    # Processing
    STRIDE = 32
