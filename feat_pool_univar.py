import pandas as pd
from scipy.stats import ttest_ind
import gbl
from sklearn.feature_selection import f_classif, f_regression, mutual_info_classif, mutual_info_regression


def t_frame_compute(frame_a, frame_b, feat_filter=[]):
    if feat_filter is not None:
        feat_list = [ft for ft in gbl.FS_feats if any(x in ft for x in feat_filter)]
    else:
        feat_list = gbl.FS_feats
    # t_test per feature
    t_frame = pd.DataFrame(index=['t_stat', 't_stat_abs', 'p_val', 'p_val_abs'], columns=feat_list)
    print(t_frame.transpose)

    for feat in feat_list:
        t_result = ttest_ind(frame_a.loc[:, feat],
                             frame_b.loc[:, feat])
        # print(t_result)
        t_frame.at['t_stat', feat] = t_result.statistic
        t_frame.at['t_stat_abs', feat] = abs(t_result.statistic)
        t_frame.at['p_val', feat] = t_result.pvalue
        t_frame.at['p_val_abs', feat] = abs(t_result.pvalue)
        # print('%s t:%f p:%f' % (feat, t_frame.loc['t_statistic', feat], t_frame.loc['p_value', feat]))

    t_frame.sort_values(by='t_stat_abs', axis=1, ascending=False, inplace=True)
    alpha = 0.05
    for column in t_frame:
        # if t_frame.loc['t_stat_abs', column] <= 1.96 or
        if t_frame.loc['p_val_abs', column] >= alpha:
            t_frame.drop(columns=column, inplace=True)
            # print('dropping %s' % column)

    return t_frame


def f_frame_compute(frame, y_tgt, task, feat_filter=[]):
    if feat_filter is not None:
        feat_list = [ft for ft in gbl.FS_feats if any(x in ft for x in feat_filter)]
    else:
        feat_list = gbl.FS_feats
    frame = frame.loc[:, feat_list]
    if task == 'clf':
        f_score, p_val = f_classif(frame, y_tgt, center=True)
    elif task == 'reg':
        f_score, p_val = f_regression(frame, y_tgt, center=True)

    f_frame = pd.DataFrame(index=['f_score', 'p_val'], columns=frame.columns.tolist())
    f_frame.at['f_score', :] = f_score/f_score.max()
    f_frame.at['p_val', :] = p_val
    print(f_frame.transpose)

    f_frame.sort_values(by='p_val', axis=1, ascending=True, inplace=True)
    alpha = 0.05
    for column in f_frame:
        if f_frame.loc['p_val', column] >= alpha:
            f_frame.drop(columns=column, inplace=True)

    return f_frame


def mi_frame_compute(frame, y_tgt, task, feat_filter=[]):
    if feat_filter is not None:
        feat_list = [ft for ft in gbl.FS_feats if any(x in ft for x in feat_filter)]
    else:
        feat_list = gbl.FS_feats
    frame = frame.loc[:, feat_list]
    if task == 'clf':
        mi = mutual_info_classif(frame, y_tgt, discrete_features='auto', n_neighbors=3, random_state=None)
    elif task == 'reg':
        mi = mutual_info_regression(frame, y_tgt, discrete_features='auto', n_neighbors=3, random_state=None)
    mi_frame = pd.DataFrame(index=['mut_info'], columns=frame.columns.tolist())
    mi_frame.at['mut_info', :] = mi/mi.max()
    print(mi_frame.transpose)

    mi_frame.sort_values(by='mut_info', axis=1, ascending=False, inplace=True)

    for column in mi_frame:
        if mi_frame.loc['mut_info', column] < 0.5:
            mi_frame.drop(columns=column, inplace=True)

    return mi_frame



