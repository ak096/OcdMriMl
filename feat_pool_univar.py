from collections import Counter
from copy import copy, deepcopy

import pandas as pd
from scipy.stats import ttest_ind
from sklearn.feature_selection import f_classif, f_regression, mutual_info_classif, mutual_info_regression

import gbl


def check_feat_filter(feat_filter):
    if not feat_filter:
        feat_list = gbl.FreeSurfer_feats
    else:
        feat_list = [ft for ft in gbl.FreeSurfer_feats if any(x in ft for x in feat_filter)]
    return feat_list


feat_max = 50


def t_frame_compute(frame_a, frame_b, feat_filter=[]):

    feat_list = check_feat_filter(feat_filter)

    # t_test per feature
    t_frame = pd.DataFrame(index=['t_stat', 't_stat_abs', 'p_val', 'p_val_abs'], columns=feat_list)

    for feat in feat_list:
        t_result = ttest_ind(frame_a.loc[:, feat],
                             frame_b.loc[:, feat])

        t_frame.at['t_stat', feat] = t_result.statistic
        t_frame.at['t_stat_abs', feat] = abs(t_result.statistic)
        t_frame.at['p_val', feat] = t_result.pvalue
        t_frame.at['p_val_abs', feat] = abs(t_result.pvalue)
        # print('%s t:%f p:%f' % (feat, t_frame.loc['t_statistic', feat], t_frame.loc['p_value', feat]))

    alpha = 0.05
    for column in t_frame:
        # if t_frame.loc['t_stat_abs', column] <= 1.96 or
        if t_frame.loc['p_val_abs', column] >= alpha or t_frame.loc['t_stat_abs', column] <= 1.96:
            t_frame.drop(columns=column, inplace=True)
            # print('dropping %s' % column)
    t_frame.sort_values(by='t_stat_abs', axis=1, ascending=False, inplace=True)
    t_feat_max = feat_max
    for column in t_frame:
        if t_frame.columns.get_loc(column) >= t_feat_max:
            t_frame.drop(columns=column, inplace=True)
    return t_frame


def f_frame_compute(frame, y_tgt, task, feat_filter=[]):

    feat_list = check_feat_filter(feat_filter)

    frame = frame.loc[:, feat_list]
    if task == 'clf':
        f_score, p_val = f_classif(frame, y_tgt)
    elif task == 'reg':
        f_score, p_val = f_regression(frame, y_tgt, center=True)

    f_frame = pd.DataFrame(index=['f_score', 'p_val'], columns=frame.columns.tolist())
    f_frame.at['f_score', :] = f_score/f_score.max()
    f_frame.at['p_val', :] = p_val

    alpha = 0.05
    for column in f_frame:
        if f_frame.loc['p_val', column] >= alpha or f_frame.loc['f_score', column] <= 0.5:
            f_frame.drop(columns=column, inplace=True)
    f_frame.sort_values(by='f_score', axis=1, ascending=False, inplace=True)
    f_feat_max = feat_max
    for column in f_frame:
        if f_frame.columns.get_loc(column) >= f_feat_max:
            f_frame.drop(columns=column, inplace=True)
    return f_frame


def mi_frame_compute(frame, y_tgt, task, feat_filter=[]):

    feat_list = check_feat_filter(feat_filter)

    frame = frame.loc[:, feat_list]
    if task == 'clf':
        mi = mutual_info_classif(frame, y_tgt, discrete_features='auto', n_neighbors=3, random_state=None)
    elif task == 'reg':
        mi = mutual_info_regression(frame, y_tgt, discrete_features='auto', n_neighbors=3, random_state=None)
    mi_frame = pd.DataFrame(index=['mut_info'], columns=frame.columns.tolist())
    mi_frame.at['mut_info', :] = mi/mi.max()

    for column in mi_frame:
        if mi_frame.loc['mut_info', column] <= 0.5:
            mi_frame.drop(columns=column, inplace=True)
    mi_frame.sort_values(by='mut_info', axis=1, ascending=False, inplace=True)
    mi_feat_max = feat_max
    for column in mi_frame:
        if mi_frame.columns.get_loc(column) >= mi_feat_max:
            mi_frame.drop(columns=column, inplace=True)
    return mi_frame


def feat_pool_compute(tgt_name, subs, feat_filter=[]):

    # compute t_feats
    if subs.resampled:  # use .iloc
        a = subs.pat_frame_train.iloc[subs.pat_names_train_bins[subs.bin_keys[0]], :]
        b = subs.pat_frame_train.iloc[subs.pat_names_train_bins[subs.bin_keys[-1]], :]
    else:  # use .loc
        a = subs.pat_frame_train.loc[subs.pat_names_train_bins[subs.bin_keys[0]], :]
        b = subs.pat_frame_train.loc[subs.pat_names_train_bins[subs.bin_keys[-1]], :]

    t_frame = t_frame_compute(a, b, feat_filter=[])  # ['thickness', 'volume'])
    t_feats = t_frame.columns.tolist()
    t_feats_num = len(t_feats)
    print('%s: computed %d T feats' % (tgt_name, t_feats_num))

    # compute f_feats
    f_frame = f_frame_compute(frame=subs.pat_frame_train, y_tgt=subs.pat_frame_train_y,
                              task=subs.tgt_task, feat_filter=[])
    f_feats = f_frame.columns.tolist()
    f_feats_num = len(f_feats)
    print('%s: computed %d F feats' % (tgt_name, f_feats_num))
    # compute mi_feats
    mi_frame = mi_frame_compute(frame=subs.pat_frame_train, y_tgt=subs.pat_frame_train_y,
                                task=subs.tgt_task, feat_filter=[])
    mi_feats = mi_frame.columns.tolist()
    mi_feats_num = len(mi_feats)
    print('%s: computed %d MI feats' % (tgt_name, mi_feats_num))

    feat_pool_all = t_feats + f_feats + mi_feats
    feat_pool_counts_frame = pd.DataFrame(index=['count'], data=dict(Counter(deepcopy(feat_pool_all))))
    feat_pool_counts_frame.sort_values(by='count', axis=1, ascending=False, inplace=True)

    feat_pool_set = list(set(feat_pool_counts_frame.columns.tolist()))

    return t_frame, f_frame, mi_frame, feat_pool_counts_frame, feat_pool_set