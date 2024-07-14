from src.data_manager import DataLoader, diamond_preprocessor
from src.trainer import Trainer
import importlib


def training_pipeline(config: dict, dataset_file_path: str, logging) -> None:
    logging.info("Training pipeline started...")

    # extract paths of the dataset and directory where to save the model
    dataset_path = dataset_file_path
    model_dir = config['models_path']

    # define the trainer
    trainer = Trainer(model_dir=model_dir)
    try:
        # load the data using a DataLoader
        logging.info(f"Loading data from {dataset_path}...")
        dataloader = DataLoader(file_path=dataset_path)
        logging.info(f"Done.")

        # load and prepare the data
        logging.info("Preparing data...")
        dataloader.preprocess(preprocess_fn=diamond_preprocessor)
        logging.info(f"Done.")

        # split into train and test
        test_size = config['test_size']
        logging.info(
            f"Splitting data into training ({int((1 - test_size) * 100)} %) and test ({int(test_size * 100)} %)...")
        x_train, x_test, y_train, y_test = dataloader.train_test_split(test_size=test_size,
                                                                       seed=config['random_seed'])

        # load the model
        module_name = "sklearn.linear_model"
        model_class = config["model"]["class"]

        # import the module
        module = importlib.import_module(module_name)

        # Retrieve the class from the module
        model_class = getattr(module, model_class)

        # initializing the model
        logging.info(f"Training model {config['model']['class']}...")
        trainer.init_model(model=model_class, params=config["model"]["params"])
        logging.info(f"Done.")

        # train the model
        trainer.train(x_train=x_train, y_train=y_train)

        # evaluate the model
        logging.info(f"Evaluating model...")
        val_metrics = trainer.evaluate(x_test=x_test, y_test=y_test, metrics=config["metrics"])
        logging.info(f"Done. Results: {val_metrics}")

        # save the model and metrics
        logging.info(f"Saving model...")
        model_path, metric_path = trainer.save(metrics=val_metrics)
        logging.info(f"Model saved to {model_path}, metrics saved to {metric_path}.")

    except Exception as e:
        logging.error(f"Error occurred: {e}")
    logging.info("Training pipeline completed!")
