from src.data_manager import DataLoader
from src.trainer import Trainer
from src.utils import load_module
from typing import Tuple
import os


def training_pipeline(config: dict, dataset_file_path: str, logging) -> Tuple[dict, str, str]:
    """
    Function representing a training pipeline that trains a model using a dataset.

    Parameters
    ----------
    config: dict
        A dictionary containing the configuration preprocessing of the training pipeline.
    dataset_file_path: str
        A string containing the path of the dataset to train.
    logging
        The logger of the training pipeline.

    Raises
    ------
    any captured exception

    Returns
    -------
    Tuple[dict, str, str]
        A tuple containing:
            - a dictionary containing the validation metrics
            - a string containing the path to the saved model
            - a string containing the path to the saved metrics
    """

    logging.info("Training pipeline started...")

    # extract paths of the dataset and directory where to save the model
    dataset_path = dataset_file_path
    model_dir = config['models_path']

    # define the trainer
    log_normalize = config["model"]["log_normalize"]
    trainer = Trainer(model_dir=model_dir, log_normalize=log_normalize)
    try:
        # load the data using a DataLoader
        logging.info(f"Loading data from {dataset_path}...")
        dataloader = DataLoader(file_path=dataset_path)
        logging.info(f"Done.")

        # preprocess the data
        preprocessor = config["preprocessing_fn"]
        logging.info(f"Preparing data with {preprocessor} preprocessor...")
        preprocessor = load_module(preprocessor)
        dataloader.preprocess(preprocess_fn=preprocessor)
        logging.info(f"Done.")

        # split into train and test
        test_size = config['test_size']
        logging.info(
            f"Splitting data into training ({int((1 - test_size) * 100)} %) and test ({int(test_size * 100)} %)...")
        x_train, x_test, y_train, y_test = dataloader.train_test_split(test_size=test_size,
                                                                       seed=config['random_seed'])

        # load the model
        model = config["model"]["class"]
        model_class = load_module(model)

        # if tuning is required
        tuning_parameters = config['model']['parameter_tuning']
        training_params = {}
        if tuning_parameters:
            # tune the model to extract the best hyperparameters
            logging.info("Tuning the model...")
            best_params = trainer.tune(x=x_train, y=y_train, model=model_class, params=config["model"]["params"],
                                       tuning=tuning_parameters)
            logging.info("Done.")
            logging.info(f"Best hyperparameters: {best_params}")

            # append the found hyperparameters to the training preprocessing
            training_params.update(best_params)

        # extract the training preprocessing from the configuration
        params = config["model"]["params"]
        if params:
            training_params.update(config["model"]["params"])

        # initialize the model
        logging.info(f"Training model {config['model']['class']}...")
        trainer.init_model(model=model_class, params=training_params)

        # train the model
        trainer.train(x_train=x_train, y_train=y_train)
        logging.info(f"Done.")

        # evaluate the model
        logging.info(f"Evaluating model...")
        val_metrics = trainer.evaluate(x_test=x_test, y_test=y_test, metrics=config["val_metrics"])
        logging.info(f"Done.")
        logging.info(f"Results: {val_metrics}")

        # save model and metrics
        logging.info(f"Saving model...")
        model_path, metric_path = trainer.save(metrics=val_metrics, preprocessor=config["preprocessing_fn"])
        logging.info(f"Model saved to {model_path}, metrics saved to {metric_path}.")

        logging.info("Training pipeline completed!")

        # return validation metrics and saved file paths
        return val_metrics, model_path, metric_path

    except Exception:
        # propagate any captured exception
        raise
