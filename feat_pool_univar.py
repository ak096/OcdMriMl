import pandas as pd
from scipy.stats import ttest_ind
import gbl
from sklearn.feature_selection import f_classif, f_regression, mutual_info_classif, mutual_info_regression


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
    f_frame.sort_values(by='f_score', axis=1, ascending=True, inplace=True)
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



