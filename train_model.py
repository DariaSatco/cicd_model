import pandas as pd
import yaml

from src.model import train_model, compute_model_metrics, inference
from src.feature_engineering import build_target

from sklearn.model_selection import train_test_split


if __name__=='__main__':
    
    with open('config.yaml') as f:
        params = yaml.safe_load(f)

    clean_df = pd.read_csv(params['preprocessed_data'])

    feat_params = params['feature_engineering']
    X = clean_df.drop(columns=feat_params['target_label_col'])
    y, lb = build_target(clean_df, label_col=feat_params['target_label_col'])
    # print(X.shape)
    # print(y.shape)

    model_params = params['modelling']
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=model_params['test_size'],
                                                        random_state=model_params['random_seed'])
    
    best_model = train_model(X_train, y_train,
                            feat_params,
                            model_params)
    
    y_train_preds = inference(best_model, X_train)
    y_test_preds = inference(best_model, X_test)

    precision_train, recall_train, fbeta_train = compute_model_metrics(y_train, y_train_preds)
    precision_test, recall_test, fbeta_test = compute_model_metrics(y_test, y_test_preds)

    print(precision_test, recall_test, fbeta_test)