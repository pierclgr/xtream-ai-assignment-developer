import pandas as pd
import os
from joblib import dump
import json
import numpy as np
import importlib
from typing import Tuple
import uuid
import random


class Trainer:
    """
    Class that represents a trainer, training a given model with the given dataset.
    """

    def __init__(self, model_dir: str) -> None:
        """
        Constructor method of the Trainer class.

        Parameters
        ----------
        model_dir (str):
            The directory containing the saved models.
        """

        self.model = None
        self.model_dir = model_dir

        # create the directory containing the saved model if it does not exists
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    def init_model(self, model, params) -> None:
        """
        Parameters
        ----------
        model:
            The model class to be trained.
        params:
            The parameters of the model to train, if any (default is None).
        """

        # define model to train given its parameters
        if params:
            self.model = model(params)
        else:
            self.model = model()

    def train(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Method that trains the model with the given data.

        Parameters
        ----------
        x_train (pd.DataFrame):
            The features of the training dataset.
        y_train (pd.Series):
            The labels of the training dataset.

        Raises
        ------
        Exception if the model to train is not initialized.

        Returns
        -------
        None
        """

        if self.model:
            y_train = np.log(y_train)
            self.model.fit(x_train, y_train)
        else:
            raise Exception("Model to train not defined.")

    def evaluate(self, x_test: pd.DataFrame, y_test: pd.Series, metrics: list) -> dict:
        """
        Method that evaluates the model with the given test set.

        Parameters
        ----------
        x_test (pd.DataFrame):
            The features of the testing set to use for the validation.
        y_test (pd.Series):
            The labels of the testing set to use for the validation.
        metrics (list):
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
            pred_log = self.model.predict(x_test)

            # compute log transformation of prediction logs
            predictions = np.exp(pred_log)

            # load the metrics
            module_name = "sklearn.metrics"

            value_metrics = {}
            # for each metric in the list
            for metric_name in metrics:
                module = importlib.import_module(module_name)

                # retrieve the class from the module
                metric_fn = getattr(module, metric_name)

                # compute the metric
                metric_value = metric_fn(y_test, predictions)

                # append to output list
                value_metrics[metric_name] = float(metric_value)

            return value_metrics
        else:
            raise Exception("Model to evaluate not defined.")

    def save(self, metrics: dict) -> Tuple[str, str]:
        """
        Method that saves the trained model and its validation metrics into two different files.

        Parameters
        ----------
        metrics (dict):
            A dictionary containing validation metrics of the model to save.

        Returns
        -------
        Tuple[str, str]:
            A tuple containing the path of the saved model and its metrics.
        """

        # generate id for the model
        id = uuid.uuid4()

        model_filename = f"{type(self.model).__name__}_{id}_model.joblib"
        metrics_filename = f"{type(self.model).__name__}_{id}_metrics.json"

        # create paths for the model file and the metrics file
        model_path = os.path.join(self.model_dir, model_filename)
        metrics_path = os.path.join(self.model_dir, metrics_filename)

        # save trained model
        dump(self.model, model_path)

        # save model validation metrics
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)

        return model_path, metrics_path
