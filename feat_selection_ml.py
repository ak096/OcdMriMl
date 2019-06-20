from random import randint

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV
#from yellowbrick.features import RFECV
#from mlxtend.frequent_patterns import apriori
#from mlxtend.preprocessing import TransactionEncoder
import pyfpgrowth

from train_predict import set_paramgrid_est, conf_interval


def grid_rfe_cv(tgt_name, est_type, task, feat_pool, X, y, cv_folds, n_min_feat=None, n_max_feat=None, params=None, scoring=None):

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
        print('%s/%s : RFECV %d/%d computed %d feats' % (tgt_name, est_type, idx, len(param_grid), len(feat_sel)))
        feat_sels_rfecv.append(feat_sel)

    return feat_sels_rfecv


def freq_item_sets_compute(dataset, min_sup=0.6):
    return [list(l) for l in pyfpgrowth.find_frequent_patterns(dataset, round(min_sup*len(dataset))).keys()]


# def freq_item_sets(dataset, min_support=0.6): #expects list of lists, returns pandas DataFrame
#     te = TransactionEncoder()
#     te_ary = te.fit(dataset).transform(dataset)
#     df = pd.DataFrame(te_ary, columns=te.columns_)
#     freq_item_sets_frame = apriori(df, min_support=min_support, use_colnames=True, verbose=1)
#     freq_item_sets_frame.sort_values(by='support', axis=0, ascending=False, inplace=True)
#     freq_item_sets_frame['length'] = freq_item_sets_frame['itemsets'].apply(lambda x: len(x))
#
#     return freq_item_sets_frame


def feat_perm_imp_compute(all_tgt_results, all_fsets_results, task):
    delete = []
    feat_all = list(set([f for f in [fs['feat_set_list'] for fs in all_fsets_results.values()]]))
    fimp_collect = {}
    for f in feat_all:
        fimp_collect[f] = []
        for tgt, est_type_dict in all_tgt_results.items():
            if task in tgt:
                for est_type, fset_dict in est_type_dict.items():
                    for fset, fset_results in fset_dict.items():
                        index = "{}_{}_{}".format(tgt, est_type, fset)
                        if not fset_results.data['pred_scores']:
                            delete.append(index)
                        elif f in fset_results.data['feat_imp_frames'][0].columns.tolist():
                            fimp_collect[f].append(
                                fset_results.data['feat_imp_frames'][0].loc['perm_imp', f])
    fimp_result = {}
    for k, v in fimp_collect.items():
        fimp_result[k] = {'freq': len(fimp_collect[k]),
                          'perm_imp_high': np.max(fimp_collect[k]),
                          'perm_imp_avg': conf_interval(fimp_collect[k])[0],
                          'perm_imp_low': np.min(fimp_collect[k]),
                          'perm_imp_ci':conf_interval(fimp_collect[k])[1]
                         }

    return fimp_result, delete
