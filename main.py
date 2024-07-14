import json
import logging
import os
from src.pipeline import training_pipeline
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import shutil


class DatasetFolderHandler(FileSystemEventHandler):
    """
    Class that handles a dataset folder and runs the training pipeline everytime the folder is updated.
    """

    def __init__(self, config: dict) -> None:
        """
        Constructor method of the DatasetFolderHandler class.

        Parameters
        ----------
        config (dict):
            The dictionary containing the settings of the training, extracted from the config file.

        Returns
        -------
        None
        """

        self.dataset_dir = config['dataset_path']
        self.last_modified = None
        self.config = config

    def on_any_event(self, event) -> None:
        """
        Method that handles any event in the directory.

        Parameters
        ----------
        event:
            The event received from the handler.

        Returns
        -------
        None
        """

        if event.is_directory:
            return None

        # if the event is either the created or modified event
        if event.event_type in ('created', 'modified'):
            current_modified = os.path.getmtime(event.src_path)
            if self.last_modified is None or current_modified != self.last_modified:
                self.last_modified = current_modified
                logging.info(f"Detected change in: {event.src_path}")

                # run the pipeline
                self.run_pipeline()

    def run_pipeline(self) -> None:
        """
        Method that runs the pipeline everytime the monitored folder changes.

        Returns
        -------
        None
        """

        self.models_history = []

        # delete the models directory if it exists exits
        model_dir = self.config['models_path']
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)

        # create the models directory
        os.makedirs(model_dir)

        try:
            for filename in os.listdir(self.dataset_dir):
                if filename.endswith('.csv'):
                    data_path = os.path.join(self.dataset_dir, filename)
                    logging.info(f"Processing file: {data_path}")
                    val_metrics, model_path, metrics_path = training_pipeline(self.config, dataset_file_path=data_path,
                                                                              logging=logging)
                    self.models_history.append({"metrics": val_metrics,
                                                "model_path": model_path,
                                                "metrics_path": metrics_path})
        except Exception as e:
            logging.error(f"Error in running pipeline: {e}")


def monitor_folder(config: dict) -> None:
    """
    Function that runs the DatasetFolderHandler.

    Parameters
    ----------
    config (dict):
        The config of the training.

    Returns
    -------
    None
    """

    event_handler = DatasetFolderHandler(config)
    event_handler.run_pipeline()
    observer = Observer()
    observer.schedule(event_handler, config["dataset_path"], recursive=True)
    observer.start()

    logging.info(f"Started monitoring folder \"{config['dataset_path']}\"...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

    # print the best model
    logging.info(f"Stopped monitoring folder.")
    max_r2_score = 0
    best_model_data = None

    # for each saved model
    for model_data in event_handler.models_history:
        r2_score = model_data['metrics']['r2_score']

        if r2_score > max_r2_score:
            max_r2_score = r2_score
            best_model_data = model_data

    logging.info("The best model is:")
    logging.info(best_model_data)


def main() -> None:
    """
    The main method of the program.

    Returns
    -------
    None
    """

    # load the configuration file
    config_path = 'config/settings.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    # define logging
    log_file = config['log_file']

    if os.path.exists(log_file):
        os.remove(log_file)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[
                            logging.FileHandler(log_file),
                            logging.StreamHandler()
                        ])

    monitor_folder(config)


if __name__ == '__main__':
    main()
