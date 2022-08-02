import numpy as np
import pandas as pd
from typing import List
import yaml

from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


def build_target(input_data: pd.DataFrame, 
                 label_col: str) -> np.array:
    """
    Transform salary column into binary target

    Args:
        input_data
        label_col

    Returns:
        array with binary encoded label
    """
    target_data = input_data[label_col]

    lb = LabelBinarizer()
    y = lb.fit_transform(target_data.values).ravel()

    return y, lb


def feature_engineering_pipeline(categorical_cols: List=[],
                                 numerical_cols: List=[]):
    """
    Build pipeline to transform features. All features put into
    categorical_list_cols are one-hot-encoded and all features 
    from numerical_cols are scaled with Standard scaler. If you
    keep lists empty, columns will be kept "as is"

    Args:
        categorical_cols
        numerical_cols

    Returns:
        ColumnTransformer object
    """

    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    scaler = StandardScaler()

    preprocessor = ColumnTransformer(
            transformers=[
                ("one_hot_enc", encoder, categorical_cols),
                ("scaler", scaler, numerical_cols)
            ],
            remainder='passthrough'
        )
    
    return preprocessor


if __name__=='main':

    with open('config.yaml') as f:
        params = yaml.safe_load(f)

    clean_df = pd.read_csv(params['preprocessed_data'])
    
    feat_params = params['feature_engineering']
    X = clean_df.drop(columns=feat_params['target_label_col'])
    feat_preproc = feature_engineering_pipeline(categorical_cols=feat_params['categorical_cols'])
    features = feat_preproc.fit_transform(X)

    y = build_target(clean_df, label_col=feat_params['target_label_col'])