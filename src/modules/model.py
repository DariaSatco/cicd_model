import numpy as np
import pandas as pd
import pickle
from typing import Dict, List

import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

from aequitas.group import Group
from aequitas.bias import Bias 
from aequitas.fairness import Fairness

from .feature_engineering import feature_engineering_pipeline


def _build_param_grid(input_params: Dict,
                     step_name: str) -> Dict:
    """
    Utility function to compile parameters names for
    sklearn pipeline GridSearchCV

    Args:
        input_params (Dict) : dictionary with parameters grid
        step_name (string)  : name of pipeline step

    Returns:
        Updated dictionary with parameters with the right name of
        keys
    """
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


def save_model(model: sklearn.base.BaseEstimator,
               path: str):
    """
    Utility function to save model with pickle
    """
    with open(path, 'wb') as file:
        pickle.dump(model, file)


def load_model(path: str):
    """
    Utility function to load saved model
    """
    with open(path, 'rb') as file:
        model = pickle.load(file)
    return model


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


def map_label(y: np.array,
              label_encoder: sklearn.base.BaseEstimator):
    """
    Map binary predictions 1/0 into original label format

    Args:
        y (array)                     : binary predictions from model predict
        label_encoder (sklearn model) : encoder of original label column into binary

    Returns:
        array with decoded labels
    """
    y_decoded = label_encoder.inverse_transform(y)
    return y_decoded


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
    

def evaluate_model_by_slices(X: pd.DataFrame, 
                             y: np.array, 
                             y_preds: np.array,
                             slicing_cols: List,
                             parity_metric: str = 'TPR'):
    """
    Evaluate model fairness by comparing parity_metric across
    different data slices formed by slicing_cols unique values

    Args:
        X (Dataframe)           : features table
        y (array)               : labels array
        y_preds (array)         : predictions array
        slicing_cols (List)     : list of column names (from X)
        parity_metric (string)  : name of metric of interest. Should be one of
                            the following: 'FDR', 'FPR', 'FOR', 'FNR', 'TPR', 
                            'TNR', 'NPV'


    Returns:
        Dataframe with disparity value and fairness outcome (True/False) vs
        major subgroup across each column from slicing_cols
    """
    df_aq = X.copy()
    df_aq['label_value'] = list(y)
    df_aq['score'] = list(y_preds)
    
    # calculating confusion matrix and corresponding metrics 
    # across different subsegments formed by feature value cuts
    g = Group()
    xtab, _ = g.get_crosstabs(df_aq, attr_cols=slicing_cols)

    # filter out small subgroups & groups with low rate of positive cases
    xtab_filtered = xtab[ (xtab['group_size']>=100) & (xtab['prev']>=0.1)].reset_index(drop=True).copy()

    # collecting disparities across slices and metrics
    # Disparity is checked versus biggest subsegment (major)
    b = Bias()
    majority_bdf = b.get_disparity_major_group(xtab_filtered, original_df=df_aq)

    # calculating fairness across different subsegments
    f = Fairness()
    fdf = f.get_group_value_fairness(majority_bdf)

    return fdf[['attribute_name', 'attribute_value', 
                parity_metric.lower()+'_disparity', 
                parity_metric+' Parity']]


