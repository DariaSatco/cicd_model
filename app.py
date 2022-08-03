from fastapi import FastAPI
from pydantic import BaseModel, Field
import yaml
import pandas as pd

import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from src.modules.model import load_model

# pull data with DVC
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI()

class Sample(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias='education-num')
    marital_status: str = Field(alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias='capital-gain')
    capital_loss: int = Field(alias='capital-loss')
    hours_per_week: int = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')


@app.get("/")
async def root():
    return {"message": "Welcome to Udacity project app!"}


@app.post("/predict/")
async def run_inference(sample: Sample):

    with open('model_config.yaml') as f:
        params = yaml.safe_load(f)

    model = load_model(params['paths']['pretrained_model'])
    label_encoder = load_model(params['paths']['label_encoder'])

    input_df = pd.DataFrame([sample.dict(by_alias=True)])

    encoded_prediction = model.predict(input_df)
    prediction = label_encoder.inverse_transform(encoded_prediction)[0]

    return prediction
    