#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.metrics import roc_auc_score, roc_curve, auc

from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

# ====================================================
# data
# 1.data StandardScaler
# ====================================================
# full data
iris = datasets.load_iris()
X, y = iris.data, iris.target
y = (y == 0).astype("int32")



# ====================================================
# create testing dataset
# ====================================================
X, X_test, y, y_test = train_test_split(X, y, test_size = 0.2, random_state = 29)


# ====================================================
# data preprocessing
# ====================================================
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.fit_transform(X_test)


# ====================================================
# basic models
# ====================================================
lr_l1 = LogisticRegression(penalty = "l1", solver = "liblinear")
lr_l1 = LogisticRegression(penalty = "l1", solver = "saga")
lr_l2 = LogisticRegression(penalty = "l2", solver = "newton-cg")
lr_l2 = LogisticRegression(penalty = "l2", solver = "lbfgs")
lr_l2 = LogisticRegression(penalty = "l2", solver = "liblinear")
lr_l2 = LogisticRegression(penalty = "l2", solver = "sag")
lr_l2 = LogisticRegression(penalty = "l2", solver = "saga")
lr_cv_l1 = LogisticRegressionCV(penalty = "l1", solver = "liblinear")
lr_cv_l1 = LogisticRegressionCV(penalty = "l1", solver = "saga")
lr_cv_l2 = LogisticRegressionCV(penalty = "l2", solver = "newton-cg")
lr_cv_l2 = LogisticRegressionCV(penalty = "l2", solver = "lbfgs")
lr_cv_l2 = LogisticRegressionCV(penalty = "l2", solver = "liblinear")
lr_cv_l2 = LogisticRegressionCV(penalty = "l2", solver = "sag")
lr_cv_l2 = LogisticRegressionCV(penalty = "l2", solver = "saga")
lr_sgd_l1 = SGDClassifier(loss = "log", penalty = "l1")
lr_sgd_l2 = SGDClassifier(loss = "log", penalty = "l2")
lr_sgd_elsasticnet = SGDClassifier(loss = "log", penalty = "elasticnet")


# ----------------------------------------------------------
# basic model
# hold-out
# ----------------------------------------------------------
# split dataset to train and validate dataset
X_train, X_val, y_train, y_val = train_test_split(X, y)

# train the lr with train dataset
lr_l1.fit(X_train, y_train)

# scores on train dataset
score_train = lr_l1.score(X_train, y_train)
print(score_train)


# prediction and scores on validate dataset
y_pred_class = lr_l1.predict(X_val)
y_pred_proba = lr_l1.predict_proba(X_val)
y_pred_log_proba = lr_l1.predict_log_proba(X_val)
print(y_pred_class)
print(y_pred_proba)
print(y_pred_log_proba)

validate_score = lr_l1.score(X_val, y_val)
print(validate_score)

# ----------------------------------------------------------
# cross_val_score, cross_validate, cross_val_predict
# ----------------------------------------------------------



# ----------------------------------------------------------
# KFold, RepeatedKFold
# ----------------------------------------------------------




# ----------------------------------------------------------
# StratifiedKFold, RepeatedStratifiedKFold, StratifiedShuffleSplit
# ----------------------------------------------------------





# ----------------------------------------------------------
# GridSearchCV, RandomizedSearchCV
# ----------------------------------------------------------
scoring = [
	"accuracy_score"
	"precision_score",
	"recall_score",
	"f1_score",
	"roc_auc_score",
	"auc"
]


param_grid = {

}

param_random = {

}


