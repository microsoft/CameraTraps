::: PW_FT_classification.main

# Main Function

## Description
Main function for training or evaluating a ResNet model (50 or 18) using PyTorch Lightning. It loads configurations, initializes the model, logger, and other components based on provided arguments.

## Arguments
- `config (str)`: Path to the configuration file.
- `project (str)`: Name of the project for logging.
- `gpus (str)`: Comma-separated GPU ids for training.
- `logger_type (str)`: Type of logger to use (wandb, comet, tensorboard, csv).
- `evaluate (str)`: Path to the model checkpoint for evaluation.
- `np_threads (str)`: Number of numpy threads to use.
- `session (int)`: Session number for logging purposes.
- `seed (int)`: Random seed for reproducibility.
- `dev (bool)`: Development mode flag.
- `val (bool)`: Validation mode flag.
- `test (bool)`: Testing mode flag.
- `predict (bool)`: Prediction mode flag.
- `predict_root (str)`: Root directory for prediction outputs.