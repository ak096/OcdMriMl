import pandas as pd
from scipy.stats import ttest_ind
import gbl


def t_frame_compute(pat_frame_train, con_frame, feat_filter):
    feat_list = [ft for ft in gbl.FS_feats if any(x in ft for x in feat_filter)]
    # t_test per feature
    t_frame = pd.DataFrame(index=['t_statistic', 't_stat_abs', 'p_value', 'p_val_abs'], columns=feat_list)
    # print(t_frame)
    for feat in feat_list:
        t_result = ttest_ind(pat_frame_train.loc[:, feat],
                             con_frame.loc[:, feat])
        # print(t_result)
        t_frame.at['t_statistic', feat] = t_result.statistic
        t_frame.at['t_stat_abs', feat] = abs(t_result.statistic)
        t_frame.at['p_value', feat] = t_result.pvalue
        t_frame.at['p_val_abs', feat] = abs(t_result.pvalue)
        # print('%s t:%f p:%f' % (feat, t_frame.loc['t_statistic', feat], t_frame.loc['p_value', feat]))
    t_frame.sort_values(by='t_stat_abs', axis=1, ascending=False, inplace=True)

    for column in t_frame:
        if t_frame.loc['t_stat_abs', column] <= 1.96 or t_frame.loc['p_val_abs', column] >= 0.05:
            t_frame.drop(columns=column, inplace=True)
            # print('dropping %s' % column)

    return t_frame


