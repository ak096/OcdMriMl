#from random import randint
#from collections import OrderedDict
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV
#from yellowbrick.features import RFECV
#from mlxtend.frequent_patterns import apriori
#from mlxtend.preprocessing import TransactionEncoder
import pyfpgrowth
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


def freq_item_sets_compute(dataset, min_sup=1.0):

    dataset = [list(np.int16(ds)) for ds in dataset] # convert to int16s to save memory avoid MemoryError

    patterns = pyfpgrowth.find_frequent_patterns(dataset, round(min_sup*len(dataset)))
    if len(patterns) > 0:
        # sets of largest size
        # for k, v in patterns.items():
        #     freq_item_sets.setdefault(len(k), []).append(list(k)) #keys are length of sets
        # max_key = max(freq_item_sets, key=int)
        # print('%d freq item sets of len %d' % (len(freq_item_sets[max_key]), max_key))
        # return freq_item_sets[max_key]
        # all maximum size unique subsets
        del_keys = []
        for k0 in patterns.keys():
            for k1 in patterns.keys():
                if set(k0) != set(k1):
                    if set(k0).issubset(set(k1)):
                        del_keys.append(k0)
        freq_item_sets = [list(s) for s in patterns.keys() if s not in del_keys]
        return freq_item_sets
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
        fpi_result[gbl.all_feat_names[k]] = {'pi_freq': len(fpis[k]),
                                             'pi_avg': round(conf_interval(fpis[k])[0], 3),
                                             'pi_high': np.max(fpis[k]),
                                             'pi_low': np.min(fpis[k]),
                                             'pi_ci': round(conf_interval(fpis[k])[1], 3)
                                             }

    return fpi_result


def largest_common_subsets(dataset, min_sup=10):
    dataset = [list(np.int16(ds)) for ds in dataset]  # convert to int16s to save memory avoid MemoryError
    lcs_list = []
    #smallest_num_sets = round(len(dataset)*min_sup)
    smallest_num_sets = min_sup
    for num in np.arange(smallest_num_sets, len(dataset)+1, 1):
        print('finding all combis of size: %d' % num)
        subset_idxs = list(combinations(np.arange(len(dataset)), num))
        print('combis found: %d' % len(subset_idxs))
        for i, sis in enumerate(subset_idxs):
            dataset_sub = [set(dataset[si]) for si in sis]
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


def cust_func(s):
    t = sum(s.freq)
    if type(s.freq) is not pd.core.series.Series: # is than non-iterable int or float
        s.freq = [s.freq]
    if type(s.perm_imp_avg) is not pd.core.series.Series: # is than non-iterable int or float
        s.perm_imp_avg = [s.perm_imp_avg]
    pi_freq = t
    pi_avg = round(sum([(i[0]/t)*i[1] for i in list(zip(s.freq, s.perm_imp_avg))])/t, 3)
    pi_high = round(np.max(s.perm_imp_high), 3)
    pi_low = round(np.min(s.perm_imp_low), 3)

    return pd.Series({'pi_freq': pi_freq, 'pi_avg': pi_avg, 'pi_high': pi_high, 'pi_low': pi_low})


def combine_fpi_frames(*args):
    fpi_all_frame = pd.concat([a for a in args])
    fpi_all_frame.apply(cust_func)
    return fpi_all_frame.sort_values(by='pi_avg', axis=1, ascending=False, inplace=True) # without ci


def combine_dicts(*args):
    super_dict = {}
    for dict in args:
        for k, v in dict.items():
            super_dict.setdefault(k, []).extend(v)
    return super_dict


def fpi_all_results_frame_compute(*args):
    super_dict = combine_dicts(args)
    fpi_all_results_dict = feat_perm_imp_compute(super_dict)
    fpi_all_results_frame = pd.DataFrame.from_dict(fpi_all_results_dict)
    return fpi_all_results_frame.sort_values(by='pi_avg', axis=1, ascending=False, inplace=True)