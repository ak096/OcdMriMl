import pandas as pd


def init_globals(frame):
    global regType_list
    regType_list = ['rfr', 'svmr', 'mlpr', 'lr', 'enr', 'rr', 'lasr', 'laslarr', 'gbr']
    global clfType_list
    clfType_list = ['rfc', 'svmc', 'mlpc', 'abc', 'logr', 'knc', 'gpc', 'gnb', 'lda', 'qda', 'gbc']
    global normType_list
    normType_list = ['std', 'minMax', 'robust']
    global FS_feats
    FS_feats = frame.columns.tolist()
    global t_frame_perNorm_list
    t_frame_perNorm_list = []

    return
