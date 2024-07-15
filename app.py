import sqlite3
import threading
import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import uvicorn
import numpy as np
from src.model_manager import ModelManager
from src.utils import load_module
from src.data_manager import diamond_preprocessor

app = FastAPI()

# create thread-local storage for SQLite connection and cursor
thread_local = threading.local()


def get_connection():
    """
    Function to get a new SQLite connection.

    Returns
    -------
    SQLite connection
    """

    if not hasattr(thread_local, "connection"):
        if not os.path.exists(config["request_database"]):
            open('requests.db', 'a').close()  # Create an empty file if it doesn't exist
        thread_local.connection = sqlite3.connect('requests.db')
    return thread_local.connection


def get_cursor():
    """
    Function to get a new SQLite cursor.

    Returns
    -------
    SQLite cursor
    """

    return get_connection().cursor()


def create_requests_table():
    """
    Function to create a table of requests in the database if it does not exist.
    Returns
    -------

    """
    cursor = get_cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS requests (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        endpoint TEXT,
                        request JSON,
                        response JSON,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );''')
    get_connection().commit()


class PredictRequest(BaseModel):
    """
    Class that represents a predict request.
    """

    carat: float
    cut: str
    color: str
    clarity: str
    depth: float
    table: float
    x: float
    y: float
    z: float


class SamplesRequest(BaseModel):
    """
    Class that represents a samples request.
    """

    cut: str
    color: str
    clarity: str
    weight: float
    n_samples: int = 5


# load configuration file
with open('config/api.json', 'r') as f:
    config = json.load(f)

# create requests table
create_requests_table()

# extract the metric to use to choose the best model
best_metric = config['best_metric']

# load the dataset
try:
    dataframe = pd.read_csv(config["dataset_path"])
except:
    raise HTTPException(status_code=500, detail="Dataset could not be loaded")

# initialize the model manager
try:
    mm = ModelManager(config)
except:
    raise HTTPException(status_code=500, detail="Model could not be loaded")


@app.post("/predict")
def predict(request: PredictRequest):
    """
    Function representing the predict endpoint.

    Parameters
    ----------
    request: PredictRequest
        The body of the predict request.

    Returns
    -------
    dict:
        The predicted value.
    """

    # load models and get the best one
    try:
        mm.load_models_and_metrics()
        model = mm.get_best_model(best_metric)
        print(model["model"], model["metrics"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")

    # preprocess the data
    features = pd.DataFrame([request.dict()])
    preprocessor_name = model['preprocessor']['preprocessor']
    log_normalize = model['preprocessor']['log_normalize']
    preprocessor = load_module(preprocessor_name)

    if "linear" in preprocessor_name:
        df = diamond_preprocessor(dataframe)
        df = preprocessor(df)
        df = df.drop(columns=["carat", "price", "x"])
        columns = df.columns
        categorical_columns = {col: [False] for col in columns}
        categorical_columns = pd.DataFrame(categorical_columns)
        categorical_columns[f'cut_{features.loc[0, "cut"]}'] = True
        categorical_columns[f'color_{features.loc[0, "color"]}'] = True
        categorical_columns[f'clarity_{features.loc[0, "clarity"]}'] = True
        features.drop(columns=["carat", "x"])
        features = pd.concat([features, categorical_columns], axis=1)
        features = preprocessor(features)
    else:
        features = preprocessor(features)

    # compute the prediction
    prediction = model["model"].predict(features)
    if log_normalize:
        prediction = np.exp(prediction)

    prediction = float(prediction[0])

    # save request and response to database
    cursor = get_cursor()
    cursor.execute('''INSERT INTO requests (endpoint, request, response) VALUES (?, ?, ?)''',
                   ('predict', json.dumps(request.dict()), json.dumps({"predicted_value": prediction})))
    get_connection().commit()

    return {"predicted_value": prediction}


@app.post("/samples")
def get_samples(request: SamplesRequest):
    """
    Function representing the get samples endpoint.

    Parameters
    ----------
    request: SamplesRequest
        The body of the get samples request.

    Returns
    -------
    dict:
        A dictionary containing the list of samples.
    """

    cut = request.cut
    color = request.color
    clarity = request.clarity
    weight = request.weight
    n_samples = request.n_samples

    # filter samples by cut, color and clarity
    filtered_df = dataframe[
        (dataframe['cut'] == cut) & (dataframe['color'] == color) & (dataframe['clarity'] == clarity)]

    if filtered_df.empty:
        raise HTTPException(status_code=404, detail="No matching samples found")

    # order extracted samples by weight difference
    filtered_df['weight_diff'] = np.abs(filtered_df['carat'] - weight)
    filtered_df = filtered_df.sort_values(by='weight_diff').head(n_samples)
    filtered_df = filtered_df.drop(columns=['weight_diff'])

    # save request and response to database
    response = filtered_df.to_dict(orient='records')
    cursor = get_cursor()
    cursor.execute('''INSERT INTO requests (endpoint, request, response) VALUES (?, ?, ?)''',
                   ('samples', json.dumps(request.dict()), json.dumps(response)))
    get_connection().commit()

    return response


@app.get("/logs")
def get_logs() -> list:
    """
    Function that returns all logs from the database.

    Returns
    -------
    list
        A list of dictionaries representing each log entry.
    """

    cursor = get_cursor()
    cursor.execute('''SELECT * FROM requests''')
    logs = cursor.fetchall()

    # convert to list of dictionaries
    logs_list = []
    for log in logs:
        log_dict = {
            "id": log[0],
            "endpoint": log[1],
            "request": json.loads(log[2]),
            "response": json.loads(log[3]),
            "timestamp": log[4]
        }
        logs_list.append(log_dict)

    return logs_list


@app.on_event("shutdown")
def shutdown_event():
    get_connection().close()


if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except KeyboardInterrupt:
        print("Application interrupted!")
