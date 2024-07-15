import json
import logging
import os
from src.pipeline import training_pipeline
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import traceback


def run_pipeline(config_path: str) -> None:
    """
    Method that runs the pipeline.

    Parameters
    ----------
    config_path: str
        The path to the config file.

    Returns
    -------
    None
    """

    # load the config
    with open(config_path, 'r') as f:
        config = json.load(f)
    dataset_dir = config["dataset_path"]

    # define logging
    log_file = config['log_file']
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[
                            logging.FileHandler(log_file),
                            logging.StreamHandler()
                        ])

    # iterate all csv files in the dataset directory and run the training pipeline for each of it
    for filename in os.listdir(dataset_dir):
        if filename.endswith('.csv'):
            data_path = os.path.join(dataset_dir, filename)
            logging.info(f"Processing file: {data_path}")
            training_pipeline(config, dataset_file_path=data_path,
                              logging=logging)


def main() -> None:
    """
    The main method of the program.

    Returns
    -------
    None
    """

    # load the configuration file
    config_path = 'config/training_pipeline.json'
    run_pipeline(config_path)


if __name__ == '__main__':
    main()
