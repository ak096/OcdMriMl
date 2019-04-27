import pandas as pd


def init_globals(con_frame):
    global regType_list
    regType_list = ['rfr', 'svmr', 'mlpr', 'lr', 'enr', 'rr', 'lasr', 'laslarr']
    global clrType_list
    clrType_list = ['rfc', 'svmc', 'mlpc', 'abc', 'logr', 'knc', 'gpc', 'gnb', 'lda', 'qda']
    global normType_list
    normType_list = ['std', 'minMax', 'robust']
    global FS_feats
    FS_feats = con_frame.columns.tolist()
    global t_frame_perNorm_list
    t_frame_perNorm_list = []

    return
