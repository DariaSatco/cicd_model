import pandas as pd
import yaml

from dvclive import Live

from modules.model import (train_model,
                           compute_model_metrics,
                           inference,
                           save_model,
                           evaluate_model_by_slices)
from modules.feature_engineering import build_target

from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    # load inputs
    with open('config.yaml') as f:
        params = yaml.safe_load(f)

    clean_df = pd.read_csv(params['paths']['preprocessed_data'])

    feat_params = params['feature_engineering']
    X = clean_df.drop(columns=feat_params['target_label_col'])
    y, lb = build_target(clean_df, label_col=feat_params['target_label_col'])

    # save label encoder
    save_model(lb, params['paths']['label_encoder'])

    model_params = params['modelling']
    # split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=model_params['test_size'],
                                                        random_state=model_params['random_seed'],
                                                        stratify=y)

    # fit model
    best_model = train_model(X_train, y_train,
                             feat_params,
                             model_params)

    # save pretrained model
    save_model(best_model, params['paths']['pretrained_model'])

    # evaluate model with basic quality scores
    y_train_preds = inference(best_model, X_train)
    y_test_preds = inference(best_model, X_test)

    precision_train, recall_train, fbeta_train = compute_model_metrics(
        y_train, y_train_preds)
    precision_test, recall_test, fbeta_test = compute_model_metrics(
        y_test, y_test_preds)

    # log metrics with DVC
    live = Live("evaluation")
    # train
    live.log("train precision", precision_train)
    live.log("train recall", recall_train)
    live.log("train f1", fbeta_train)

    # test
    live.log("test precision", precision_test)
    live.log("test recall", recall_test)
    live.log("test f1", fbeta_test)

    # show values in std output
    print(
        'Train:',
        'precision=',
        precision_train,
        'recall=',
        recall_train,
        'f1=',
        fbeta_train)
    print(
        'Test:',
        'precision=',
        precision_test,
        'recall=',
        recall_test,
        'f1=',
        fbeta_test)

    # evaluate model fairness
    eval_params = params['evaluation']
    disparity_tab = evaluate_model_by_slices(
        X_test,
        y_test,
        y_test_preds,
        slicing_cols=eval_params['slicing_cols'],
        parity_metric=eval_params['parity_metric'])
