from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import uvicorn
import json
import numpy as np
from src.model_manager import ModelManager
from src.utils import load_module
from src.data_manager import diamond_preprocessor


class PredictRequest(BaseModel):
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
    cut: str
    color: str
    clarity: str
    weight: float
    n_samples: int = 5


app = FastAPI()

# Load configuration
with open('config/api.json', 'r') as f:
    config = json.load(f)

best_metric = config['best_metric']
try:
    dataframe = pd.read_csv(config["dataset_path"])
except:
    raise HTTPException(status_code=500, detail="Dataset could not be loaded")

try:
    mm = ModelManager(config)
except:
    raise HTTPException(status_code=500, detail="Model could not be loaded")


@app.post("/predict")
def predict(request: PredictRequest):
    # get the best model
    try:
        mm.load_models_and_metrics()
        model = mm.get_best_model(best_metric)
        print(model["model"], model["metrics"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")

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

    prediction = model["model"].predict(features)
    if log_normalize:
        prediction = np.exp(prediction)
    return {"predicted_value": float(prediction[0])}


@app.post("/samples")
def get_samples(request: SamplesRequest):
    cut = request.cut
    color = request.color
    clarity = request.clarity
    weight = request.weight
    n_samples = request.n_samples

    # Filter dataset by cut, color, and clarity
    filtered_df = dataframe[
        (dataframe['cut'] == cut) & (dataframe['color'] == color) & (dataframe['clarity'] == clarity)]

    if filtered_df.empty:
        raise HTTPException(status_code=404, detail="No matching samples found")

    # Compute weight difference and sort
    filtered_df['weight_diff'] = np.abs(filtered_df.loc[:, 'carat'] - weight)
    filtered_df = filtered_df.sort_values(by='weight_diff').head(n_samples)

    # Drop the temporary 'weight_diff' column
    filtered_df = filtered_df.drop(columns=['weight_diff'])

    return filtered_df.to_dict(orient='records')


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
