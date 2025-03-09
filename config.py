"""
Configuration parameters for the health prediction model.
"""

# Data Parameters
DATA_PATH = "data/"
TRAIN_TEST_SPLIT = 0.2
TIME_STEPS = 4  # Number of time steps to consider for sequence prediction
PREDICTION_HORIZON = 1  # Days ahead to predict

# Feature Parameters
FEATURES = ["temperature", "spo2", "heart_rate"]
TARGET = "health_status"
TIMESTAMP_COL = "timestamp"

# Model Parameters
MODEL_TYPE = "lstm"  # Options: "lstm", "gru", "mlp"
HIDDEN_UNITS = [64, 32]
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

# Fine-tuning Parameters
FINE_TUNE_EPOCHS = 20
FINE_TUNE_LEARNING_RATE = 0.0005
TRANSFER_LAYERS = 1  # Number of layers to fine-tune (from the top)

# Paths
MODEL_SAVE_PATH = "models/health_prediction_model.h5"
SCALER_SAVE_PATH = "models/feature_scaler.pkl"
MODEL_HISTORY_PATH = "models/model_history.json"

# Evaluation Metrics
METRICS = ["accuracy", "precision", "recall", "f1_score", "roc_auc"] 