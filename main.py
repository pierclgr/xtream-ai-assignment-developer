import json
import logging
import os
from src.pipeline import training_pipeline
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import traceback


class DatasetFolderHandler(FileSystemEventHandler):
    """
    Class that handles a dataset folder and runs the training pipeline everytime the folder is updated.
    """

    def __init__(self, config_path: str) -> None:
        """
        Constructor method of the DatasetFolderHandler class.

        Parameters
        ----------
        config_path: str
            The path of the config file.

        Returns
        -------
        None
        """

        self.last_modified = None
        self.config_path = config_path

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

        # load the config
        with open(self.config_path, 'r') as f:
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

        for filename in os.listdir(dataset_dir):
            if filename.endswith('.csv'):
                data_path = os.path.join(dataset_dir, filename)
                logging.info(f"Processing file: {data_path}")
                training_pipeline(config, dataset_file_path=data_path,
                                  logging=logging)


def monitor_folder(config_path: str) -> None:
    """
    Function that runs the DatasetFolderHandler.

    Parameters
    ----------
    config: str
        The path of the config of the training.

    Returns
    -------
    None
    """
    observer = Observer()
    with open(config_path, 'r') as f:
        config = json.load(f)
    dataset_dir = config["dataset_path"]

    event_handler = DatasetFolderHandler(config_path)
    observer.schedule(event_handler, dataset_dir, recursive=True)
    observer.start()

    try:
        event_handler.run_pipeline()

        logging.info(f"Started monitoring folder \"{config['dataset_path']}\"...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Requested exiting...")
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        logging.error(f"Traceback: {traceback.print_tb(e.__traceback__)}")

    observer.stop()
    observer.join()

    # print the best model
    logging.info(f"Stopped monitoring folder.")


def main() -> None:
    """
    The main method of the program.

    Returns
    -------
    None
    """

    # load the configuration file
    config_path = 'config/training_pipeline.json'

    monitor_folder(config_path)


if __name__ == '__main__':
    main()
