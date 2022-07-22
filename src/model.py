import numpy as np
from typing import Dict

import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

from .feature_engineering import feature_engineering_pipeline


def _build_param_grid(input_params: Dict,
                     step_name: str) -> Dict:
    
    output_params = {}
    for key in input_params:
        output_params[step_name + '__' + key] = input_params[key]
    
    return output_params


def train_model(X_train: np.array, 
                y_train: np.array,
                feat_engineering_params: Dict,
                modelling_params: Dict):
    """
    Trains a machine learning model and returns it.
    
    Args:
        X_train (np.array) : Training data
        y_train (np.array) : Labels
    
    Returns:
        Trained machine learning model
    """

    # feature engineering pipeline
    feature_engineering = feature_engineering_pipeline(
        categorical_cols=feat_engineering_params['categorical_cols']
        )

    # combine feature engineering and estimator into single pipeline
    model_pipe = Pipeline(
        steps=[
            ("feature_engineering", feature_engineering),
            (
                "random_forest", RandomForestClassifier(
                    random_state=modelling_params['random_seed']
                )
            ),
        ]
    )

    # organize grid search to tune hyper parameters
    params_grid = _build_param_grid(modelling_params['rf_params_grid'], 'random_forest')
    
    search = GridSearchCV(model_pipe, 
                          params_grid, 
                          cv=modelling_params['cv'], 
                          n_jobs=4)
    search.fit(X_train, y_train)

    return search.best_estimator_


def compute_model_metrics(y: np.array, preds: np.array):
    """
    Validates the trained machine learning model using precision, recall, and F1.
    
    Args:
        y  (np.array) : known labels, binarized
        preds (np.array) : predicted labels, binarized
    
    Returns:
        precision (float)
        recall (float)
        fbeta  (float)
    """

    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    
    return precision, recall, fbeta


def inference(model: sklearn.base.BaseEstimator, 
              X: np.array):
    """ 
    Run model inferences and return the predictions.
    
    Args:
        model (BaseEstimator) : trained machine learning model
        X (np.array) : data used for prediction
    
    Returns:
        preds (np.array) : predictions from the model
    """
    preds = model.predict(X)
    return preds