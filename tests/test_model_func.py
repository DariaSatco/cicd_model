import pandas as pd
from typing import Dict
from ..src.modules.feature_engineering import feature_engineering_pipeline, build_target

def test_column_names(data: pd.DataFrame):
    """
    Test that all data columns are in place
    """
    expected_colums = [
        'age', 
        'workclass', 
        'fnlgt', 
        'education', 
        'education-num',
        'marital-status', 
        'occupation', 
        'relationship', 
        'race', 
        'sex',
        'capital-gain', 
        'capital-loss', 
        'hours-per-week', 
        'native-country',
        'salary'
    ]

    data_columns = data.columns.values

    # Enforce same column names in same order
    assert list(expected_colums) == list(data_columns)


def test_label_without_nan(data: pd.DataFrame, 
                           params: Dict):
    """
    Check that there is no NaN values in target column
    """
    assert data[params['feature_engineering']['target_label_col']].isnull().sum() == 0


def test_binary_label(data: pd.DataFrame,
                      params: Dict):
    """
    Check that there are only 2 unique values in 
    target column to assure correct functionality 
    of build_target function
    """
    y, lb = build_target(data, params['feature_engineering']['target_label_col'])
    assert len(set(lb.classes_)) == 2


def test_cat_to_num_feat_transform(data: pd.DataFrame,
                                   params: Dict):
    """
    Test feature preprocessing and assert that all features
    were transformed into numeric values
    """
    feat_params = params['feature_engineering']
    X = data.drop(columns=feat_params['target_label_col'])
    feat_preproc = feature_engineering_pipeline(categorical_cols=feat_params['categorical_cols'])
    features = feat_preproc.fit_transform(X)
    assert features.dtype.name in ['float32', 'float64']
