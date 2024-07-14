import pandas as pd
import os
from joblib import dump
import json
import numpy as np
from src.utils import load_module
from typing import Tuple
import uuid
import optuna
from sklearn.model_selection import train_test_split


class Trainer:
    """
    Class that represents a trainer, training a given model with the given dataset.
    """

    def __init__(self, model_dir: str, log_normalize: bool = True) -> None:
        """
        Constructor method of the Trainer class.

        Parameters
        ----------
        model_dir: str
            The directory containing the saved models.
        log_normalize: bool
            A boolean to indicate whether to log normalize the labels in training/validation or not.

        Returns
        -------
        None
        """

        self.model = None
        self.model_dir = model_dir
        self.log_normalize = log_normalize

        # create the directory containing the saved model if it does not exists
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    def tune(self, x: pd.DataFrame, y: pd.Series, model, params: dict, tuning: dict) -> dict:
        """
        Method that tunes the hyperparameters of the model, returning the best set of preprocessing found.

        Parameters
        ----------
        x: pd.DataFrame
            The training features to use for the tuning.
        y: pd.DataFrame
            The training labels to use for the tuning.
        model:
            The model to tune.
        params: dict
            The preprocessing of the model to tune.
        tuning: dict
            The hyperparameters to tune.

        Returns
        -------
        dict:
            A dictionary containing the set of best found hyperparameters.
        """

        def optimize(trial: optuna.trial.Trial) -> float:
            """
            Function to optimize when doing the hyperparmeter tuning.

            Parameters
            ----------
            trial: optuna.trial.Trial
                The trial to train.

            Returns
            -------
            float:
                The metric to optimize.
            """

            tuning_params = {}

            # extract all the hyperparameters of the tuning
            tuning_metric = tuning["metric"]
            seed = tuning["random_seed"]
            val_perc = tuning["val_perc"]

            for param, value in tuning.items():
                if isinstance(value, dict):
                    if value["type"] == "float":
                        tuning_params[param] = trial.suggest_float(name=param, low=value["min"], high=value["max"],
                                                                   log=True)
                    elif value["type"] == "int":
                        tuning_params[param] = trial.suggest_int(name=param, low=value["min"], high=value["max"])
                    elif value["type"] == "categorical":
                        tuning_params[param] = trial.suggest_categorical(name=param, choices=value["values"])

            tuning_params.update(params) if params else tuning_params

            # split the training dataset into training and validation
            x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=val_perc, random_state=seed)

            # train the model
            self.init_model(model=model, params=tuning_params)
            self.train(x_train, y_train)

            # make predictions
            value_metrics = self.evaluate(x_test=x_val, y_test=y_val, metrics=[tuning_metric])

            return value_metrics[tuning_metric]

        # initialize the tuning study and execute it
        study = optuna.create_study(direction='minimize', study_name='model tuning')
        study.optimize(optimize, n_trials=tuning["n_trials"])

        # return the best hyperparameters found
        return study.best_params

    def init_model(self, model, params: dict = None) -> None:
        """
        Method that initializes the model.

        Parameters
        ----------
        model:
            The model class to initialize.
        params: dict
            The parameter of the model to initialize, if any (default is None)

        Returns
        -------
        None
        """

        # define model to train given its preprocessing
        if not params:
            params = {}
        self.model = model(**params)

    def train(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Method that trains the model with the given data.

        Parameters
        ----------
        x_train: pd.DataFrame
            The features of the training dataset.
        y_train: pd.Series
            The labels of the training dataset.

        Raises
        ------
        Exception if the model to train is not initialized.

        Returns
        -------
        None
        """

        if self.model:
            if self.log_normalize:
                y_train = np.log(y_train)
            self.model.fit(x_train, y_train)
        else:
            raise Exception("Model to train not defined.")

    def evaluate(self, x_test: pd.DataFrame, y_test: pd.Series, metrics: list) -> dict:
        """
        Method that evaluates the model with the given test set.

        Parameters
        ----------
        x_test: pd.DataFrame
            The features of the testing set to use for the validation.
        y_test: pd.Series
            The labels of the testing set to use for the validation.
        metrics: list
            A list containing the metrics functions to use for the validation.

        Raises
        ------
        Exception if the model to evaluate is not initialized.

        Returns
        -------
        dict:
            A dict containing all the results of the given metrics for the validation
        """

        if self.model:
            # compute predictions
            predictions = self.model.predict(x_test)

            # compute log transformation of prediction logs
            if self.log_normalize:
                predictions = np.exp(predictions)

            value_metrics = {}
            # for each metric in the list
            for metric_name in metrics:
                # retrieve the class from the module
                metric_fn = load_module(metric_name)

                # compute the metric
                metric_value = metric_fn(y_test, predictions)

                # append to output list
                value_metrics[metric_name] = float(metric_value)

            return value_metrics
        else:
            raise Exception("Model to evaluate not defined.")

    def save(self, metrics: dict, preprocessor: str) -> Tuple[str, str]:
        """
        Method that saves the trained model and its validation metrics into two different files.

        Parameters
        ----------
        metrics: dict
            A dictionary containing validation metrics of the model to save.
        preprocessor: dict
            The used preprocessor.

        Returns
        -------
        Tuple[str, str]:
            A tuple containing the path of the saved model and its metrics.
        """

        # generate id for the model
        id = uuid.uuid4()

        model_filename = f"{type(self.model).__name__}_{id}.joblib"
        metrics_filename = f"{type(self.model).__name__}_{id}.json"

        # create paths for the model file and the metrics file
        model_weight_folder = os.path.join(self.model_dir, "weights")
        if not os.path.exists(model_weight_folder):
            os.makedirs(model_weight_folder)
        model_path = os.path.join(model_weight_folder, model_filename)

        model_metrics_folder = os.path.join(self.model_dir, "metrics")
        if not os.path.exists(model_metrics_folder):
            os.makedirs(model_metrics_folder)
        metrics_path = os.path.join(model_metrics_folder, metrics_filename)

        preprocessing_folder = os.path.join(self.model_dir, "preprocessing")
        if not os.path.exists(preprocessing_folder):
            os.makedirs(preprocessing_folder)
        preprocessing_path = os.path.join(preprocessing_folder, metrics_filename)

        # save trained model
        dump(self.model, model_path)

        # save model validation metrics
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)

        # save preprocessor
        with open(preprocessing_path, 'w') as f:
            json.dump({"preprocessor": preprocessor, "log_normalize": self.log_normalize}, f)

        return model_path, metrics_path
