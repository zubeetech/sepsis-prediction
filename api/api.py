from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel


app = FastAPI()


class sepsis_features(BaseModel):
    log_plasma: float
    log_bt1: float
    log_pressure: float
    log_bt2: float
    log_bt3: float
    log_age: float
    insurance: float
    log_bmi: float
    log_bt4: float


@app.get('/')
def root():
    return {"Status": "API is Online ..."}


forest_pipeline = joblib.load('../models/RandomForest.joblib')
ada_pipeline = joblib.load('../models/Adaboost.joblib')
logreg_pipeline = joblib.load('../models/LogReg.joblib')
encoder = joblib.load('../models/encoder.joblib')


@app.post('/rf_predict')
def predict_sepsis_rf(data: sepsis_features):
    df = pd.DataFrame([data.model_dump()])
    prediction = forest_pipeline.predict(df)
    int_features = int(prediction[0])
    label = encoder.inverse_transform([int_features])[0]
    probability = round(float(forest_pipeline.predict_proba(df)[0][int_features]*100, 2))
    final_prediction = {"prediction": label, "probability": probability}
    return {"final_prediction": final_prediction}


@app.post('/lr_predict')
def predict_sepsis_lr(data: sepsis_features):
    df = pd.DataFrame([data.model_dump()])
    prediction = logreg_pipeline.predict(df)
    int_features = int(prediction[0])
    label = encoder.inverse_transform([int_features])[0]
    probability = round(float(logreg_pipeline.predict_proba(df)[0][int_features]*100, 2))
    final_prediction = {"prediction": label, "probability": probability}
    return {"final_prediction": final_prediction}


@app.post('/ad_predict')
def predict_sepsis_lr(data: sepsis_features):
    df = pd.DataFrame([data.model_dump()])
    prediction = ada_pipeline.predict(df)
    int_features = int(prediction[0])
    label = encoder.inverse_transform([int_features])[0]
    probability = round(float(ada_pipeline.predict_proba(df)[0][int_features]*100, 2))
    final_prediction = {"prediction": label, "probability": probability}
    return {"final_prediction": final_prediction}