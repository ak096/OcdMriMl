import pandas as pd

regType_list = ['rfr', 'svmr', 'mlpr', 'lr', 'enr', 'rr', 'lasr', 'laslarr']
clrType_list = ['rfc', 'svmc', 'mlpc', 'abc', 'logr', 'knc', 'gpc', 'gnb', 'lda', 'qda']
normType_list = ['std', 'minMax', 'robust']
t_frame_perNorm_list = []
FS_feats = pd.DataFrame.columns

# variables to save to disk (pickle) ---------
# the iteration that needs to be done next
iteration = {'n': 0, 'num_tfeats': 1}
# expert-picked-feature-based models for regression and classification
hoexter_reg_models_all = []
hoexter_clr_models_all = []
boedhoe_reg_models_all = []
boedhoe_clr_models_all = []

# t_test-picked-feature-based models for regression and classification
t_reg_models_all = []
t_clr_models_all = []
# -----------

