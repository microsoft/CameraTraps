::: PW_FT_classification.src.algorithms.plain

# Plain Algorithm

## Description
Defines the architecture for training a model using PyTorch Lightning. This class inherits from PyTorch Lightning's LightningModule and sets up the model, optimizers, and training/validation/testing steps for the training process.

## Methods
- `__init__`: Initializes the Plain model.
- `configure_optimizers`: Configures the optimizers and learning rate schedulers.
- `on_train_start`: Hook function called at the start of training.
- `training_step`: Training step for each batch.
- `on_validation_start`: Hook function called at the start of validation.
- `validation_step`: Validation step for each batch.
- `on_validation_epoch_end`: Hook function called at the end of the validation epoch.
- `on_test_start`: Hook function called at the start of testing.
- `test_step`: Test step for each batch.
- `on_test_epoch_end`: Hook function called at the end of the test epoch.
- `on_predict_start`: Hook function called at the start of prediction.
- `predict_step`: Prediction step for each batch.
- `on_predict_epoch_end`: Hook function called at the end of the predict epoch.
- `eval_logging`: Logs evaluation metrics such as accuracy.