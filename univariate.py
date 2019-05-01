import pandas as pd
from scipy.stats import ttest_ind
import glob


def t_compute(pat_frame_train, con_frame, n):
    # t_test per feature
    t_frame = pd.DataFrame(index=['t_statistic', 'p_value'], columns=glob.FS_feats)
    # print(t_frame)
    for feat in glob.FS_feats:
        t_result = ttest_ind(pat_frame_train.loc[:, feat],
                             con_frame.loc[:, feat])
        # print(t_result)
        t_frame.at['t_statistic', feat] = t_result.statistic
        t_frame.at['p_value', feat] = t_result.pvalue
        # print('%s t:%f p:%f' % (feat, t_frame.loc['t_statistic', feat], t_frame.loc['p_value', feat]))
    t_frame.sort_values(by='t_statistic', axis=1, ascending=False, inplace=True)
    glob.t_frame_perNorm_list[n] = t_frame

    for column in t_frame:
        if abs(t_frame.loc['t_statistic', column]) <= 1.96 or abs(t_frame.loc['p_value', column]) >= 0.05:
            t_frame.drop(columns=column, inplace=True)
            # print('dropping %s' % column)

    return t_frame