from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel


app = FastAPI()


class sepsis_features(BaseModel):
    plasma: float
    bt1: float
    pressure: float
    bt2: float
    bt3: float
    age: float
    insurance: str
    bmi: float
    bt4: float


@app.get('/')
def root():
    return {"Status": "API is Online ..."}


forest_pipeline = joblib.load('../api/models/RandomForest.joblib')
ada_pipeline = joblib.load('../api/models/Adaboost.joblib')
logreg_pipeline = joblib.load('../api/models/LogReg.joblib')
encoder = joblib.load('../api/models/encoder.joblib')


@app.post('/rf_predict')
def predict_sepsis_rf(data: sepsis_features):
    df = pd.DataFrame([data.model_dump()])
    rf_prediction = forest_pipeline.predict(df)
    int_features = int(rf_prediction[0])
    label = encoder.inverse_transform([int_features])[0]
    probability = round(float(forest_pipeline.predict_proba(df)[0][int_features])*100, 2)
    final_prediction = {"prediction": label, "probability": probability}
    return {"final_prediction": final_prediction}


@app.post('/lr_predict')
def predict_sepsis_lr(data: sepsis_features):
    df = pd.DataFrame([data.model_dump()])
    prediction = logreg_pipeline.predict(df)
    int_features = int(prediction[0])
    label = encoder.inverse_transform([int_features])[0]
    probability = round(float(logreg_pipeline.predict_proba(df)[0][int_features])*100, 2)
    final_prediction = {"prediction": label, "probability": probability}
    return {"final_prediction": final_prediction}


@app.post('/ad_predict')
def predict_sepsis_lr(data: sepsis_features):
    df = pd.DataFrame([data.model_dump()])
    prediction = ada_pipeline.predict(df)
    int_features = int(prediction[0])
    label = encoder.inverse_transform([int_features])[0]
    probability = round(float(ada_pipeline.predict_proba(df)[0][int_features])*100, 2)
    final_prediction = {"prediction": label, "probability": probability}
    return {"final_prediction": final_prediction}