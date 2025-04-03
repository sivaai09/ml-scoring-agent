from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from io import BytesIO

app = FastAPI()

def score_model(model, X, y):
    y_pred = model.predict(X)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y, y_pred))),
        "MAE": float(mean_absolute_error(y, y_pred)),
        "R2 Score": float(r2_score(y, y_pred))
    }

@app.post("/score")
async def score_endpoint(model_file: UploadFile = File(...), test_data: UploadFile = File(...)):
    model = joblib.load(BytesIO(await model_file.read()))
    df = pd.read_csv(BytesIO(await test_data.read()))
    y = df.iloc[:, -1]
    X = df.iloc[:, :-1]
    return score_model(model, X, y)
