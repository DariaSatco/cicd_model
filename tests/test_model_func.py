import pandas as pd
from typing import Dict

from ..src.modules.feature_engineering import feature_engineering_pipeline, build_target
from ..src.modules.model import (load_model, 
                                inference, 
                                evaluate_model_by_slices, 
                                compute_model_metrics)


def test_column_names(train_data: pd.DataFrame):
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

    data_columns = train_data.columns.values

    # Enforce same column names in same order
    assert list(expected_colums) == list(data_columns)


def test_label_without_nan(train_data: pd.DataFrame,
                           params: Dict):
    """
    Check that there is no NaN values in target column
    """
    assert train_data[params['feature_engineering']
                ['target_label_col']].isnull().sum() == 0


def test_binary_label(train_data: pd.DataFrame,
                      params: Dict):
    """
    Check that there are only 2 unique values in
    target column to assure correct functionality
    of build_target function
    """
    y, lb = build_target(
        train_data, params['feature_engineering']['target_label_col'])
    assert len(set(lb.classes_)) == 2


def test_cat_to_num_feat_transform(train_data: pd.DataFrame,
                                   params: Dict):
    """
    Test feature preprocessing and assert that all features
    were transformed into numeric values
    """
    feat_params = params['feature_engineering']
    X = train_data.drop(columns=feat_params['target_label_col'])
    feat_preproc = feature_engineering_pipeline(
        categorical_cols=feat_params['categorical_cols'])
    features = feat_preproc.fit_transform(X)
    assert features.dtype.name in ['float32', 'float64']


def test_model_metrics_pass_threshold(test_data: pd.DataFrame,
                                      params: Dict):
    """
    Test that current version of the model successfully passes minimum
    quality conditions (see thresholds in config file)
    """
    X_test = test_data.drop(columns=params['feature_engineering']['target_label_col'])

    model = load_model(params['paths']['pretrained_model'])
    label_encoder = load_model(params['paths']['label_encoder'])

    y_test = label_encoder.transform(test_data[params['feature_engineering']['target_label_col']]).flatten()
    y_test_preds = inference(model, X_test)
    
    precision_test, recall_test, fbeta_test = compute_model_metrics(y_test, y_test_preds)
    assert precision_test >= params['evaluation']['threshold']['precision']
    assert recall_test >= params['evaluation']['threshold']['recall']
    assert fbeta_test >= params['evaluation']['threshold']['fbeta']


def test_model_is_fair(test_data: pd.DataFrame,
                       params: Dict):
    """
    Test that current version of the model is fair "enough" (see threshold
    condition in config)
    """
    X_test = test_data.drop(columns=params['feature_engineering']['target_label_col'])

    model = load_model(params['paths']['pretrained_model'])
    label_encoder = load_model(params['paths']['label_encoder'])

    y_test = label_encoder.transform(test_data[params['feature_engineering']['target_label_col']]).flatten()
    y_test_preds = inference(model, X_test)

    eval_params = params['evaluation']
    disparity_tab = evaluate_model_by_slices(
        X_test,
        y_test,
        y_test_preds,
        slicing_cols=eval_params['slicing_cols'],
        parity_metric=eval_params['parity_metric'])

    avg_parity = disparity_tab[[eval_params['parity_metric']+' Parity']].mean().values[0]
    assert avg_parity >= params['evaluation']['threshold']['parity']
