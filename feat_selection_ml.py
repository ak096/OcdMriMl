from sklearn.feature_selection import RFECV
#from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
#from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.svm import LinearSVC, LinearSVR

from xgboost import XGBClassifier, XGBRegressor

import pandas as pd

#from yellowbrick.features import RFECV

from random import randint

import numpy as np

from train_predict import set_paramgrid_est

from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder


def grid_rfe_cv(est_type, task, feat_pool, X, y, cv_folds, n_min_feat=None, n_max_feat=None, params=None, scoring=None):

    if n_min_feat is not None:
        if len(feat_pool) <= n_min_feat:
            return feat_pool
    param_grid, est = set_paramgrid_est(est_type, task)

    # if params is not None:
    #     est.set_params(**params)
    feat_sels_rfecv = []
    for idx, grid_point in enumerate(param_grid):
        grid_point.update(params)
        est.set_params(**grid_point)

        sel = RFECV(est, min_features_to_select=n_min_feat, cv=cv_folds, n_jobs=-1, step=1, verbose=0, scoring=scoring)
        sel.fit(X.loc[:, feat_pool], y)

        feat_sel = [feat_pool[i] for i in np.where(sel.support_)[0]]
        feat_sel.sort()
        print('rfecv returns %d feats' % len(feat_sel))
        feat_sels_rfecv.append(feat_sel)

    return feat_sels_rfecv


def freq_item_sets(dataset, min_support=0.6): #expects list of lists, returns pandas DataFrame
    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    freq_item_sets_frame = apriori(df, min_support=min_support, use_colnames=True)
    freq_item_sets_frame.sort_values(by='support', axis=0, ascending=False, inplace=True)
    freq_item_sets_frame['length'] = freq_item_sets_frame['itemsets'].apply(lambda x: len(x))

    return freq_item_sets_frame