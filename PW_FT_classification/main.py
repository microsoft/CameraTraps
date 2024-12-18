# %%
# Importing libraries
import os
import yaml
import typer
from munch import Munch
# %%
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, CometLogger, TensorBoardLogger, WandbLogger
# %%
from src import algorithms
from src import datasets
# %%
from src.utils import batch_detection_cropping
from src.utils import data_splitting

app = typer.Typer(pretty_exceptions_short=True, pretty_exceptions_show_locals=False)
# %%
@app.command()
def main(
        config:str='./configs/config.yaml',
        project:str='Custom-classification',
        gpus:str='0', 
        logger_type:str='csv',
        evaluate:str=None,
        np_threads:str='32',
        session:int=0,
        seed:int=0,
        dev:bool=False,
        val:bool=False,
        test:bool=False,
        predict:bool=False,
        predict_root:str=""
    ):
    """
    Main function for training or evaluating a ResNet model (50 or 18) using PyTorch Lightning.
    It loads configurations, initializes the model, logger, and other components based on provided arguments.

    Args:
        config (str): Path to the configuration file.
        project (str): Name of the project for logging.
        gpus (str): Comma-separated GPU ids for training.
        logger_type (str): Type of logger to use (wandb, comet, tensorboard, csv).
        evaluate (str): Path to the model checkpoint for evaluation.
        np_threads (str): Number of numpy threads to use.
        session (int): Session number for logging purposes.
        seed (int): Random seed for reproducibility.
        dev (bool): Development mode flag.
        val (bool): Validation mode flag.
        predict (bool): Prediction mode flag.
        predict_root (str): Root directory for prediction outputs.
    """

    # GPU configuration: set up GPUs based on availability and user specification
    gpus = gpus if torch.cuda.is_available() else None
    gpus = [int(i) for i in gpus.split(',')]

    # Environment variable setup for numpy multi-threading. It is important to avoid cpu and ram issues.
    os.environ["OMP_NUM_THREADS"] = str(np_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(np_threads)
    os.environ["MKL_NUM_THREADS"] = str(np_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(np_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(np_threads)
    # Load and set configurations from the YAML file
    with open(config) as f:
        conf = Munch(yaml.load(f, Loader=yaml.FullLoader))
    conf.evaluate = evaluate
    conf.val = val
    conf.test = test
    conf.predict = predict
    conf.predict_root = predict_root

    # Set a global seed for reproducibility
    pl.seed_everything(seed)

    # If the annotation directory does not have a data split, split the data first
    if conf.split_data:
        # Replace annotation dir from config with the directory containing the split files
        conf.annotation_dir = os.path.dirname(conf.split_path)
        # Split the data according to the split type
        if conf.split_type == 'location':
            data_splitting.split_by_location(conf.split_path, conf.annotation_dir, conf.test_size, conf.val_size)
        elif conf.split_type == 'sequence':
            data_splitting.split_by_seq(conf.split_path, conf.annotation_dir, conf.test_size, conf.val_size)
        elif conf.split_type == 'random':
            data_splitting.create_splits(conf.split_path, conf.annotation_dir, conf.test_size, conf.val_size)
        else:
            raise ValueError('Invalid split type: {}. Available options: random, location, sequence.'.format(conf.split_type))
        
    if not conf.predict:
        # Get the path to the annotation files, and we only want to do this if we are not predicting
        if conf.test:
            test_annotations = os.path.join(conf.dataset_root, 'test_annotations.csv')
            # Crop test data (most likely we don't need this)
            batch_detection_cropping.batch_detection_cropping(conf.dataset_root, os.path.join(conf.dataset_root, "cropped_resized"), test_annotations)
        else:
            train_annotations = os.path.join(conf.dataset_root, 'train_annotations.csv')
            val_annotations = os.path.join(conf.dataset_root, 'val_annotations.csv')
            # Crop training data
            batch_detection_cropping.batch_detection_cropping(conf.dataset_root, os.path.join(conf.dataset_root, "cropped_resized"), train_annotations)
            # Crop validation data
            batch_detection_cropping.batch_detection_cropping(conf.dataset_root, os.path.join(conf.dataset_root, "cropped_resized"), val_annotations)

    # Dataset and algorithm loading based on the configuration
    dataset = datasets.__dict__[conf.dataset_name](conf=conf)
    learner = algorithms.__dict__[conf.algorithm](conf=conf,
                                                  train_class_counts=dataset.train_class_counts, 
                                                  id_to_labels=dataset.id_to_labels)

    # Logger setup based on the specified logger type
    log_folder = 'log_dev' if dev else 'log'
    logger = None
    if logger_type == 'csv':
        logger = CSVLogger(
            save_dir='./{}/{}/{}'.format(log_folder, conf.log_dir, conf.algorithm),
            prefix=project,
            name='{}_{}'.format(conf.algorithm, conf.conf_id),
            version=session
        )
    elif logger_type == 'tensorboard':
        logger = TensorBoardLogger(
            save_dir='./{}/{}/{}'.format(log_folder, conf.log_dir, conf.algorithm),
            prefix=project,
            name='{}_{}'.format(conf.algorithm, conf.conf_id),
            version=session
        )
    elif logger_type == 'comet':
        logger = CometLogger(
            api_key=os.environ.get("COMET_API_KEY"),
            save_dir='./{}/{}/{}'.format(log_folder, conf.log_dir, conf.algorithm),
            project_name=project, 
            experiment_name='{}_{}_{}'.format(conf.algorithm, conf.conf_id, session),
        )
    elif logger_type == 'wandb':
        logger = WandbLogger(
            save_dir='./{}/{}/{}'.format(log_folder, conf.log_dir, conf.algorithm),
            project=project,  
            name='{}_{}_{}'.format(conf.algorithm, conf.conf_id, session),
        )

    # Callbacks for model checkpointing and learning rate monitoring
    weights_folder = 'weights_dev' if dev else 'weights'
    checkpoint_callback = ModelCheckpoint(
        monitor='valid_mac_acc', mode='max', dirpath='./{}/{}/{}'.format(weights_folder, conf.log_dir, conf.algorithm),
        save_top_k=1, filename='{}-{}'.format(conf.conf_id, session) + '-{epoch:02d}-{valid_mac_acc:.2f}', verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Trainer configuration in PyTorch Lightning
    trainer = pl.Trainer(
        max_epochs=conf.num_epochs,
        check_val_every_n_epoch=1, 
        log_every_n_steps = conf.log_interval, 
        accelerator='gpu',
        devices=gpus,
        logger=None if evaluate is not None else logger,
        callbacks=[lr_monitor, checkpoint_callback],
        strategy='auto',
        num_sanity_val_steps=0,
        profiler=None
    )
    # Training, validation, or evaluation execution based on the mode
    if evaluate is not None:
        if val:
            trainer.validate(learner, dataloaders=[dataset.val_dataloader()], ckpt_path=evaluate)
        elif predict:
            trainer.predict(learner, dataloaders=[dataset.predict_dataloader()], ckpt_path=evaluate)
        elif test:
            trainer.test(learner, dataloaders=[dataset.test_dataloader()], ckpt_path=evaluate)
        else:
            print('Invalid mode for evaluation.')
    else:
        trainer.fit(learner, datamodule=dataset)
# %%
if __name__ == '__main__':
    app()

# %%
