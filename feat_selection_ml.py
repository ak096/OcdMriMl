#from random import randint
#from collections import OrderedDict
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV
#from yellowbrick.features import RFECV
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
import pyfpgrowth
from orangecontrib.associate.fpgrowth import frequent_itemsets

import gbl
#from sys import getsizeof

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
        sel.fit(X.loc[:, feat_pool].values, y.values)

        feat_sel = [feat_pool[i] for i in np.where(sel.support_)[0]]
        print('%s/%s: RFECV %d/%d computed %d feats' % (tgt_name, est_type, idx+1, len(param_grid), len(feat_sel)))
        # for a, b, c in zip(feat_sel[::3], feat_sel[1::3], feat_sel[2::3]):
        #     print('{:<30}{:<30}{:<}'.format(a, b, c))
        feat_sels_rfecv.append(feat_sel)

    return feat_sels_rfecv


def compute_fqis_fpgrowth_dict(super_ilists, min_sup=0.6):
    print('computing fpgrowth')
    super_ilists = [list(np.int16(ds)) for ds in super_ilists] # convert to int16s to save memory avoid MemoryError

    patterns = pyfpgrowth.find_frequent_patterns(super_ilists, round(min_sup * len(super_ilists)))
    return patterns
    # if len(patterns) > 0:
    #     # sets of largest size
    #     # for k, v in patterns.items():
    #     #     freq_item_sets.setdefault(len(k), []).append(list(k)) #keys are length of sets
    #     # max_key = max(freq_item_sets, key=int)
    #     # print('%d freq item sets of len %d' % (len(freq_item_sets[max_key]), max_key))
    #     # return freq_item_sets[max_key]
    #     # all maximum size unique subsets
    #     del_keys = []
    #     for k0 in patterns.keys():
    #         for k1 in patterns.keys():
    #             if set(k0) != set(k1):
    #                 if set(k0).issubset(set(k1)):
    #                     del_keys.append(k0)
    #     freq_item_sets = [list(s) for s in patterns.keys() if s not in del_keys]
    #     return freq_item_sets
    # else:
    #     return dataset


def compute_fqis_apriori_frame(super_ilists, min_sup=0.6): # expects list of lists, returns pandas DataFrame
    print('computing apriori')
    te = TransactionEncoder()
    te_ary = te.fit(super_ilists).transform(super_ilists)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    freq_item_sets_frame = apriori(df, min_support=min_sup, use_colnames=True, verbose=2)
    freq_item_sets_frame.sort_values(by='support', axis=0, ascending=False, inplace=True)
    freq_item_sets_frame['length'] = freq_item_sets_frame['itemset'].apply(lambda x: len(x))

    return freq_item_sets_frame


def compute_fqis_fpgrowth_orange3_list(super_ilists, min_sup=0.6):
    print('computing fpgrowth orange')
    itemsets = frequent_itemsets(super_ilists, min_sup)
    return list(itemsets)


def largest_common_subsets(super_ilists, min_sup=10):
    super_ilists = [list(np.int16(ds)) for ds in super_ilists]  # convert to int16s to save memory avoid MemoryError
    lcs_list = []
    #smallest_num_sets = round(len(dataset)*min_sup)
    smallest_num_sets = min_sup
    for num in np.arange(smallest_num_sets, len(super_ilists) + 1, 1):
        print('finding all combis of size: %d' % num)
        subset_idxs = list(combinations(np.arange(len(super_ilists)), num))
        print('combis found: %d' % len(subset_idxs))
        for i, sis in enumerate(subset_idxs):
            dataset_sub = [set(super_ilists[si]) for si in sis]
            lcs = set.intersection(*dataset_sub)
            #print('combi: %d/%d lcs size: %d' % (i, len(subset_idxs)-1, len(lcs)))
            if len(lcs):
                if list(lcs) not in lcs_list:
                    lcs_list.append(list(lcs))
                    print('updating lcs list with lcs of size: %d' % len(lcs))
                else:
                    print('skipping lcs list')
    if not lcs_list:
        print('lcs list is empty!!!!!!!!!!!!!!!!!!')
    return lcs_list


def compute_lcs_dict(super_ilists, min_sup=0.5):
    super_isets_list = [set(np.int16(sil)) for sil in super_ilists]  # convert to int16s to save memory avoid MemoryError
    lcs_dict = {}
    min_num_sets = round(len(super_isets_list) * min_sup)
    #min_num_sets = min_sup
    for num in np.arange(min_num_sets, len(super_isets_list) + 1, 1):
        print('finding all combis of size: %d' % num)
        siss_idxs = list(combinations(np.arange(len(super_isets_list)), num))
        print('combis found: %d' % len(siss_idxs))
        for i, siss_idx in enumerate(siss_idxs):
            siss_sub_list = [super_isets_list[si] for si in siss_idx]
            lcs = set.intersection(*siss_sub_list)
            #print('combi: %d/%d lcs size: %d' % (i, len(siss_idxs)-1, len(lcs)))
            if len(lcs):
                lcs_dict.setdefault(lcs, []).append(num)
    lcs_dict_max = {k: np.max(v) for k, v in lcs_dict.items()}
    return lcs_dict_max