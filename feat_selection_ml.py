#from sklearn.feature_selection import RFECV
#from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
#from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.svm import LinearSVC, LinearSVR

from xgboost import XGBClassifier, XGBRegressor

import pandas as pd

from yellowbrick.features import RFECV

from random import randint


def rfe_cv(task, feat_pool, X, y, cv_folds):

    if task is 'reg':
        estimators = [LinearSVR(), XGBRegressor()]
    elif task is 'clf':
        estimators = [LinearSVC(), XGBClassifier()]
    else:
        print('please specify task type {"reg"|"clf"}')
        return None
    feat_select = []
    for est in estimators:
        sel = RFECV(est, min_features_to_select=10, cv=cv_folds, n_jobs=-1, step=1, verbose=2)
        sel.fit(X.loc[:, feat_pool], y)
        feat_select.append(feat_pool[sel.support_])
    return feat_select[0], feat_select[1]



