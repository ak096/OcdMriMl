from random import randint
from collections import OrderedDict
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV
#from yellowbrick.features import RFECV
#from mlxtend.frequent_patterns import apriori
#from mlxtend.preprocessing import TransactionEncoder
import pyfpgrowth
import gbl
from sys import getsizeof

from train_predict import set_paramgrid_est, conf_interval


def grid_rfe_cv(tgt_name, est_type, task, feat_pool, X, y, cv_folds, n_min_feat=1, n_max_feat=None, params=None,
                scoring=None):

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
        print('%s/%s: RFECV %d/%d computed %d feats' % (tgt_name, est_type, idx+1, len(param_grid), len(feat_sel)))
        # for a, b, c in zip(feat_sel[::3], feat_sel[1::3], feat_sel[2::3]):
        #     print('{:<30}{:<30}{:<}'.format(a, b, c))
        feat_sels_rfecv.append(feat_sel)

    return feat_sels_rfecv


def freq_item_sets_compute(dataset, min_sup=1.0):
    freq_item_sets = {}
    type(dataset[0][0])
    getsizeof(dataset[0][0])
    patterns = pyfpgrowth.find_frequent_patterns(dataset, round(min_sup*len(dataset)))
    if len(patterns) > 0:
        for k, v in patterns.items():
            freq_item_sets.setdefault(len(k), []).append(list(k)) #keys are length of sets
        max_key = max(freq_item_sets, key=int)
        print('%d freq item sets of len %d' % (len(freq_item_sets[max_key]), max_key))
        return freq_item_sets[max_key]
    else:
        return dataset

# def freq_item_sets(dataset, min_support=0.6): #expects list of lists, returns pandas DataFrame
#     te = TransactionEncoder()
#     te_ary = te.fit(dataset).transform(dataset)
#     df = pd.DataFrame(te_ary, columns=te.columns_)
#     freq_item_sets_frame = apriori(df, min_support=min_support, use_colnames=True, verbose=1)
#     freq_item_sets_frame.sort_values(by='support', axis=0, ascending=False, inplace=True)
#     freq_item_sets_frame['length'] = freq_item_sets_frame['itemsets'].apply(lambda x: len(x))
#
#     return freq_item_sets_frame


def feat_perm_imp_compute(fpis):
    fpi_result = {}
    for k, v in fpis.items():
        fpi_result[gbl.all_feat_names[k]] = {'freq': len(fpis[k]),
                                             'perm_imp_high': np.max(fpis[k]),
                                             'perm_imp_avg': conf_interval(fpis[k])[0],
                                             'perm_imp_low': np.min(fpis[k]),
                                             'perm_imp_ci': conf_interval(fpis[k])[1]
                                             }

    return fpi_result
