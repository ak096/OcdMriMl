import pandas as pd


def init_globals():
    global regType_list
    regType_list = ['rfr', 'svmr', 'mlpr', 'lr', 'enr', 'rr', 'lasr', 'laslarr']
    global clrType_list
    clrType_list = ['rfc', 'svmc', 'mlpc', 'abc', 'logr', 'knc', 'gpc', 'gnb', 'lda', 'qda']
    global normType_list
    normType_list = ['std', 'minMax', 'robust']
    global t_frame_perNorm_list
    t_frame_perNorm_list = []
    global FS_feats
    FS_feats = pd.DataFrame.columns
    
    # variables to save to disk (pickle) ---------
    # the iteration that needs to be done next
    global iteration
    iteration = {'n': 0, 't_feats_num': 1, 'clr_targets': ['obs_class_3_score_range',
                                                           'com_class_3_score_range',
                                                           'YBOCS_class_3_score_range',
                                                           'obs_class_3_equal_pat',
                                                           'com_class_3_equal_pat',
                                                           'YBOCS_class_3_equal_pat']
                }
    # expert-picked-feature-based models for regression and classification
    global hoexter_reg_models_all
    hoexter_reg_models_all = []
    global hoexter_clr_models_all
    hoexter_clr_models_all = []
    global boedhoe_reg_models_all
    boedhoe_reg_models_all = []
    global boedhoe_clr_models_all
    boedhoe_clr_models_all = []
    
    # t_test-picked-feature-based models for regression and classification
    global t_reg_models_all
    t_reg_models_all = []
    global t_clr_models_all
    t_clr_models_all = []
    # -----------

