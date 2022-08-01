import pandas as pd
import yaml

from src.model import (train_model, 
                       compute_model_metrics, 
                       inference, 
                       evaluate_model_by_slices)
from src.feature_engineering import build_target

from sklearn.model_selection import train_test_split

if __name__=='__main__':
    
    # load inputs
    with open('config.yaml') as f:
        params = yaml.safe_load(f)

    clean_df = pd.read_csv(params['preprocessed_data'])

    feat_params = params['feature_engineering']
    X = clean_df.drop(columns=feat_params['target_label_col'])
    y, lb = build_target(clean_df, label_col=feat_params['target_label_col'])

    model_params = params['modelling']
    # split train/test 
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=model_params['test_size'],
                                                        random_state=model_params['random_seed'])
    
    # fit model
    best_model = train_model(X_train, y_train,
                            feat_params,
                            model_params)
    
    # evaluate model with basic quality scores
    y_train_preds = inference(best_model, X_train)
    y_test_preds = inference(best_model, X_test)

    precision_train, recall_train, fbeta_train = compute_model_metrics(y_train, y_train_preds)
    precision_test, recall_test, fbeta_test = compute_model_metrics(y_test, y_test_preds)

    print('Train:', 'precision=', precision_train, 'recall=', recall_train, 'f1=', fbeta_train)
    print('Test:', 'precision=', precision_test, 'recall=', recall_test, 'f1=', fbeta_test)

    # evaluate model fairness
    eval_params = params['evaluation']
    disparity_tab = evaluate_model_by_slices(X_test, y_test, y_test_preds,
                                            slicing_cols = eval_params['slicing_cols'],
                                            parity_metric = eval_params['parity_metric'])
    