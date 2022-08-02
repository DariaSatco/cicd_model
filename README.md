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

Within the project context we don't have preference over precision or recall, this is why we are interested in maximizing F1 score. Moving forward, if there are specific use cases how to treat individuals based on predictions, we can be more interested to minimize amount of positive cases we misclassify (to maximize precision), or to increase amount of positive cases we are able to catch (to maximize recall).

### Caveats and Recommendations

We've chosen Random Forest classifier to solve the problem and used features as is with only basic transformation from categorical to numerical format. Present model has satisfactory quality and can be further improved by playing with feature engineering process and testing different architectures of classifier itself. Since model quality is not the focus of the project we leave it up to you to further enhance the model.  
