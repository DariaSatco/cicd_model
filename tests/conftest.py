import pytest
import pandas as pd
import yaml

@pytest.fixture(scope='session')
def train_data():
    with open('model_config.yaml') as f:
        params = yaml.safe_load(f)
    data_path = params['paths']['preprocessed_data']
    df = pd.read_csv(data_path)
    return df


@pytest.fixture(scope='session')
def test_data():
    with open('model_config.yaml') as f:
        params = yaml.safe_load(f)
    data_path = params['paths']['test_data']
    df = pd.read_csv(data_path)
    return df


@pytest.fixture(scope='session')
def params():
    with open('model_config.yaml') as f:
        params = yaml.safe_load(f)
    return params