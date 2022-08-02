from fastapi.testclient import TestClient
import simplejson

# Import app from app.py
from ..app import app

# Instantiate the testing client with our app
client = TestClient(app)

def test_api_get_root():
    """
    Test root get output
    """
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Welcome to Udacity project app!"}


def test_api_post_predict_known_sample():
    """
    Test that model inference output has the right format 
    and returns predicted outcome label
    """
    sample = {'age': 29,
                'workclass': 'Private',
                'fnlgt': 236436,
                'education': 'HS-grad',
                'education-num': 9,
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

    assert r.status_code == 200
    assert r.json() in ['<=50K', '>50K']


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