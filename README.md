# Deploying a Machine Learning Model on Heroku with FastAPI

This project is part of ML DevOps Udacity nanodegree. It includes an example of an ML model pipeline, where the following tools being practiced:
* [DVC](https://dvc.org) to enable data version control and remote storage with AWS S3
* [FastAPI](https://fastapi.tiangolo.com) to build a micro-service based on deployed model
* [Heroku](https://www.heroku.com) to host Python App

## Model Card

Here we put details about model built to predict whether person's income exceeds $50K/yr based on census data.

### Model Details

**Author**: Daria Satco, dasha.shatco@gmail.com \
**Date**: July 2022 \
**Version**: 0.0.1 \
**Packages & tools**:
* `scikit-learn`
* `aequitas`
* `fastapi`

### Objective

The objective of this project is to build ML model using DevOps principles. We intend to include the following steps:
 * create unit tests to monitor the model performance on various slices of the data
 * deploy the model using the FastAPI package and create API tests 
 * incorporate slice-validation and the API tests into a CI/CD framework using GitHub Actions

### Training Data

We train binary classification model to predict individual's income (1 if >$50K/yr, 0 otherwise) based on demographical features of individual extracted from Census database. To find more details on the dataset, visit this [link](https://archive.ics.uci.edu/ml/datasets/census+income). 

### Evaluation Data

Initial dataset includes 32561 samples. We split data into train and test (with 80/20 ratio by default, but also can be adjusted in `config.yaml`) and evaluate model performance on test subset.

Note: since we are dealing with unbalanced data (24% of 1's and 76% of 0's), we are stratifying train/test split across target value, to keep same proportions of 0/1 in train and test.

### Metrics

We consider model quality across 3 metrics:
* precision
* recall
* F1-score

Metrics are loged by DVC into `evaluation.json`. See example of the output below:
```json
{
    "train precision": 0.538830297219559,
    "train recall": 0.895855472901169,
    "train f1": 0.6729195769307523,
    "test precision": 0.5485714285714286,
    "test recall": 0.884479092841956,
    "test f1": 0.6771568095496473
}
```

Within the project context we don't have preference over precision or recall, this is why we are interested in maximizing F1 score. Moving forward, if there are specific use cases how to treat individuals based on predictions, we can be more interested to minimize amount of positive cases we misclassify (to maximize precision), or to increase amount of positive cases we are able to catch (to maximize recall).

### Caveats and Recommendations

We've chosen Random Forest classifier to solve the problem and used features as is with only basic transformation from categorical to numerical format. Present model has satisfactory quality and can be further improved by playing with feature engineering process and testing different architectures of classifier itself. Since model quality is not the focus of the project we leave it up to you to further enhance the model.  


## Deployment details

We set up CI/CD pipeline with the help of:
* Github Actions: CI (continuous integration) includes automatic run of unit tests with pytest after each push to main branch. Check CI steps in `.github/workflows/python-app.yml`, which includes: 
    - set up of AWS connection
    - pull of data from AWS S3 remote storage with DVC
    - linting with `flake8`
    - run of tests from `tests/test_model_func.py` with `pytest`
* Deployment of the App with Heroku: CD (continuous deployment) is being set up via integration of Heroku app with Github repo and automatic deployment, which includes:
    - set up of Python and DVC buildpacks
    - set up of AWS connection
    - pull of data from AWS S3 remote storage with DVC
    - app launch with `uvicorn`

## Sources
* [Starter pack](https://github.com/udacity/nd0821-c3-starter-code/tree/master/starter) from Udacity
* [Get started DVC guide](https://dvc.org/doc/start) 
* [Heroku App repo](https://dashboard.heroku.com/apps/census-predict-model)
* App URL: https://census-predict-model.herokuapp.com/

## How to run the app locally

* Set up environment with conda using `environment.yml` or with `requirements.txt`
* Pull data with DVC by running: `dvc pull`. See below the list of files tracked, which you'll get after `pull` done:
![list of files tracked](/imgs/dvcdag.png)
* Start the app: `uvicorn app:app --reload`. In my case it is running on: http://127.0.0.1:8000
* Check API docs via http://127.0.0.1:8000/docs
![API docs](/imgs/example.png)

or see GET and POST examples in `test_heroku_app.ipynb`:

![GET](/imgs/live-get.png)
![POST](/imgs/live-post.png)