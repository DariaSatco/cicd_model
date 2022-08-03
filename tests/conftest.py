import pytest
import pandas as pd
import yaml

@pytest.fixture(scope='session')
def data():
    with open('model_config.yaml') as f:
        params = yaml.safe_load(f)
    data_path = params['paths']['preprocessed_data']
    df = pd.read_csv(data_path)
    return df


@pytest.fixture(scope='session')
def params():
    with open('model_config.yaml') as f:
        params = yaml.safe_load(f)
    return params