import os
import joblib
import json


class ModelManager:
    def __init__(self, config):
        self.models_folder = os.path.join(config["models_path"], "weights")
        self.metrics_folder = os.path.join(config["models_path"], "metrics")
        self.preprocessor_folder = os.path.join(config["models_path"], "preprocessing")
        self.models = []

        self.load_models_and_metrics()

    def load_models_and_metrics(self):
        models_and_metrics = []

        # iterate through the files in the directory
        for filename in os.listdir(self.models_folder):
            # extract the base name before the file extension
            base_name = filename.rsplit('.')[0]

            # build the path of the model file
            model_path = os.path.join(self.models_folder, filename)

            # build the name of the metrics file
            metrics_filename = base_name + '.json'
            metrics_path = os.path.join(self.metrics_folder, metrics_filename)

            # build preprocessor path
            preprocessor_path = os.path.join(self.preprocessor_folder, metrics_filename)

            # check if the corresponding metrics file exists
            if os.path.exists(metrics_path) and os.path.exists(preprocessor_path):
                # load the model
                model = joblib.load(model_path)

                # load the metrics
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)

                # load the preprocessor
                with open(preprocessor_path, 'r') as f:
                    preprocessor = json.load(f)

                models_and_metrics.append(
                    {"model": model, "metrics": metrics, "preprocessor": preprocessor})

        self.models = models_and_metrics

    def get_best_model(self, score_name: str):
        best_score = 0
        best_model = None
        for model in self.models:
            if model["metrics"][score_name] > best_score:
                best_score = model["metrics"][score_name]
                best_model = model

        if best_model:
            return best_model
        else:
            raise Exception("No trained model available")
