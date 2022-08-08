from fastapi.testclient import TestClient
import simplejson

import sys
import os
sys.path.append(os.path.realpath(os.curdir))

# Import app from app.py
from app import app

# Instantiate the testing client with our app
client = TestClient(app)


def test_api_get_root():
    """
    Test root get output
    """
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Welcome to Udacity project app!"}


def test_api_post_predict_negative_sample():
    """
    Test that model inference output has the right format
    and returns predicted outcome label  '<=50'
    """
    sample = {'age': 17,
            'workclass': 'Private',
            'fnlgt': 41865,
            'education': '10th',
            'education-num': 6,
            'marital-status': 'Never-married',
            'occupation': 'Other-service',
            'relationship': 'Own-child',
            'race': 'White',
            'sex': 'Female',
            'capital-gain': 0,
            'capital-loss': 0,
            'hours-per-week': 25,
            'native-country': 'United-States'
            }
    body = simplejson.dumps(sample)

    r = client.post("/predict/", data=body)

    assert r.status_code == 200
    assert r.json() == '<=50K'


def test_api_post_predict_positive_sample():
    """
    Test that model inference output has the right format
    and returns predicted outcome label  '>50'
    """
    sample = {'age': 47,
            'workclass': 'Private',
            'fnlgt': 102308,
            'education': 'Masters',
            'education-num': 14,
            'marital-status': 'Married-civ-spouse',
            'occupation': 'Exec-managerial',
            'relationship': 'Husband',
            'race': 'White',
            'sex': 'Male',
            'capital-gain': 99999,
            'capital-loss': 0,
            'hours-per-week': 50,
            'native-country': 'United-States'
            }
    body = simplejson.dumps(sample)

    r = client.post("/predict/", data=body)

    assert r.status_code == 200
    assert r.json() == '>50K'



def test_api_post_predict_wrong_input():
    """
    Test validation of json input
    """
    sample = {'age': 29,
              'workclass': 'Private',
              'fnlgt': 236436,
              'education': 'HS-grad',
              'education-num': 'WRONG INPUT HERE',
              'marital-status': 'Never-married',
              'occupation': 'Adm-clerical',
              'relationship': 'Not-in-family',
              'race': 'White',
              'sex': 'Female',
              'capital-gain': 0,
              'capital-loss': 0,
              'hours-per-week': 40,
              'native-country': 'United-States'
              }
    body = simplejson.dumps(sample)

    r = client.post("/predict/", data=body)

    assert r.status_code == 422
    assert r.json()['detail'][0]['msg'] == 'value is not a valid integer'
