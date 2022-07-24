import pytest
import pandas as pd
import yaml

@pytest.fixture(scope='session')
def data():
    data_path = 'data/census_cln.csv'
    df = pd.read_csv(data_path)
    return df


@pytest.fixture(scope='session')
def params():
    with open('config.yaml') as f:
        params = yaml.safe_load(f)
    return params