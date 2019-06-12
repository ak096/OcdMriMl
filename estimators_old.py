from sklearn.svm import SVR
import gbl
import time

import numpy as np
import xgboost as xgb
from numpy.random import uniform, randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.svm import SVR

import gbl

C = np.arange(1, 1001, 99)

n_iter = 100


def svm_hyper_param_space(est_class):
    svm_hps = {'kernel': ['linear', 'rbf'],
               'C': C,
               'gamma': ['auto'],
               'probability': [True]
               }

    if est_class == 'reg':
        svm_hps['epsilon'] = [0.3, 0.5, 0.7, 0.9]
    if est_class == 'clf':
        svm_hps['class_weight'] = ['balanced']
    return svm_hps


def xgb_hyper_param_space(est_class):
    return {  #learning task parameters : objective, eval, ...
              #'objective': ,
              #'eval_meas': ,
              #booster tree parameters
              #1.set init values, comp less expensive for initial look
              #for highly imbalanced classes
              'scale_pos_weight': 1,
              #2.most impact these two
              'min_child_weight': randint(1,6),  #default:1
              "max_depth": randint(3, 8), # default:6
              #3.carry on with gamma
              "min_split_loss": uniform(0, 0.3, 0.1),  # alias:gamma, default:0 should be tuned according to loss function
              #4.these two around 0.8
              "subsample": uniform(0.5, 0, 9, 0.05),  # default
              'colsample_bytree': uniform(0.5, 0.9, 0.05),
              #5.regularization parameters : model complexity and performance and under/over fitting?
              'reg_lambda': [1], #alias:lambda (L2), default:1
              'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05], #alias:alpha (L1), default:0
              #6.decrease learning rate and increase number of trees
              "learning_rate": [0.1, 0.2],  # alias:eta, default:0.3
              'n_estimators': np.arange(100, 701, 100),  # default:100
            }

# def mlp_hyper_param_space(num_samples, est_type):
#
#     mlp_hps = {'hidden_layer_sizes': [[int(num_samples*1.5), int(num_samples*0.7)],
#                                       #[int(num_samples*2), int(num_samples)],
#                                       [int(num_samples*2.5), int(num_samples*1.3)]],
#                'activation': ['logistic', 'relu'],
#                #'learning_rate': ['constant'],
#                'solver': ['adam'],
#                #'momentum': [0.6, 0.7, 0.8, 0.9],
#                'momentum': [0.9],
#                'warm_start': [True],
#                'early_stopping': [True],
#                #'power_t': [0.3, 0.4, 0.5, 0.6, 0.7],
#                #'max_iter': np.arange(150, 301, 50),
#                'max_iter': [500],
#                #'beta_1': [0.6, 0.7, 0.8, 0.9],
#                'beta_1': [0.5],
#                #'beta_2': [0.85, 0.89, 0.95, 0.999],
#                'batch_size': [10]}
#
#     if est_type == 'reg':
#         mlp_hps['activation'].append('tanh')
#
#     return mlp_hps


# return best regressors in a list with order rfr, svmr, mlpr, lr, enr, rr, lasr, laslarr
def regress(feat_frame_train, y_train, cv_folds, performance_metric, normIdx_train, task_name, num_feats_total, t_feats_idx_subset=[]):

    t0 = time.time()

    num_samples = feat_frame_train.shape[0]
    num_feats = feat_frame_train.shape[1]
    reg_params = []
    reg_all = []

    alpha = np.arange(1, 5)*0.1

    # Random Forest Regression
    #rfr_hyper_param_space = {'n_estimators': np.arange(50, 201, 50)}
    #reg_params.append([RandomForestRegressor(), rfr_hyper_param_space, gbl.regType_list[0]])


    # SVM Regression
    svmr_hyper_param_space = svm_hyper_param_space('reg')
    reg_params.append([SVR(cache_size=2000, max_iter=10000), svmr_hyper_param_space, gbl.regType_list[1]])


    # MLP Regression
    # mlpr_hyper_param_space = mlp_hyper_param_space(num_samples, 'reg')
    # reg_params.append([MLPRegressor(), mlpr_hyper_param_space, glob.regType_list[2]])


    # # Linear Regression
    # lr_hyper_param_space = {}
    # reg_params.append([LinearRegression(), lr_hyper_param_space, glob.regType_list[3]])
    #
    #
    # # ElasticNet Regression
    # # enr_hyper_param_space = {'l1_ratio': np.arange(0, 11)*0.1,
    # #                          'max_iter': [2000],
    # #                          'precompute': ['auto']}
    # # reg_params.append([ElasticNet(), enr_hyper_param_space, glob.regType_list[4]])
    #
    # # Ridge Regression
    # rr_hyper_param_space = {'alpha': alpha}
    # reg_params.append([Ridge(), rr_hyper_param_space, glob.regType_list[5]])
    #
    #
    # # Lasso Regression
    # lasr_hyper_param_space = {'alpha': alpha}
    # reg_params.append([Lasso(), lasr_hyper_param_space, glob.regType_list[6]])
    #
    #
    # # LassoLARS Regression
    # laslarr_hyper_param_space = {'alpha': alpha}
    # reg_params.append([LassoLars(), laslarr_hyper_param_space, glob.regType_list[7]])

    # Gradient Boosting Regressor (Tree Based)
    # gbr_hyper_param_space = {
    #                          'loss': ['ls','lad','huber'],
    #                          "learning_rate": [0.05, 0.1, 0.15],
    #                          #"min_samples_split": np.linspace(0.1, 0.5, 12),
    #                          #"min_samples_leaf": np.linspace(0.1, 0.5, 12),
    #                          "max_depth": [3, 5, 7],
    #                          "max_features": ["log2", "sqrt"],
    #                          "criterion": ["friedman_mse"],
    #                          "subsample": [0.5, 0.8, 0.9, 1.0],
    #                          'n_estimators': np.arange(100, 501, 100)
    #                         }
    #
    # reg_params.append([GradientBoostingRegressor(), gbr_hyper_param_space, glob.regType_list[8]])

    # XG Boost Regressor
    xgbr_hyper_param_space = xgb_hyper_param_space()
    reg_params.append([xgb.XGBRegressor(), xgbr_hyper_param_space, gbl.regType_list[9]])

    for reg_p in reg_params:
        print("%s: Running RandomizedSearchCV with %s: %d OF %d FEATS" % (task_name, reg_p[2], num_feats, num_feats_total))
        if reg_p[2] in ['gbr', 'xgbr']:
            scoring = None
        else:
            scoring = performance_metric
        reg_all.append([
                        RandomizedSearchCV(reg_p[0], param_distributions=reg_p[1], n_iter=n_iter, n_jobs=-1, scoring=scoring,
                                           cv=cv_folds, verbose=1)
                       .fit(feat_frame_train, y_train.iloc[:, 0]),
                        reg_p[2],
                        normIdx_train,
                        num_feats,
                        t_feats_idx_subset
                        ])
    # print("REGRESS() TOOK %.2f SEC" % (time.time()-t0))
    return reg_all


# return best classifiers in a list with order rfc, svmc, mlpc, abc, logr, knc, gpc, gnb, lda, qda
def classify(feat_frame_train, y_train, cv_folds, performance_metric, normIdx_train, task_name, num_feats_total, t_feats_idx_subset=[]):

    t0 = time.time()
    num_samples = feat_frame_train.shape[0]
    num_feats = feat_frame_train.shape[1]
    clf_params = []
    clf_all = []

    # Random Forest Classification
    # rfc_hyper_param_space = {'n_estimators': np.arange(50, 201, 50),
    #                          'warm_start': [True],
    #                          'class_weight': ['balanced', 'balanced_subsample']
    #                          }
    # clf_params.append([RandomForestClassifier(), rfc_hyper_param_space, gbl.clfType_list[0]])

    # SVM Classification
    svmc_hyper_param_space = svm_hyper_param_space('clf')
    clf_params.append([SVC(cache_size=2000, max_iter=10000), svmc_hyper_param_space, gbl.clfType_list[1]])

    # MLP Classification
    # mlpc_hyper_param_space = mlp_hyper_param_space(num_samples, 'clf')
    # clf_params.append([MLPClassifier(), mlpc_hyper_param_space, glob.clfType_list[2]])

    # Adaboost Classification
    # abc_hyper_param_space = {'n_estimators': np.arange(50, 101, 10),
    #                         'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.15, 0.2, 0.25, 0.3],
    #                         'class_weight': ['balanced']}
    # clf_params.append([AdaBoostClassifier(), abc_hyper_param_space, glob.clfType_list[3]])

    # Logistic Regression
    # logr_hyper_param_space = {
    #                           'penalty': ['l2'],
    #                           'multi_class': ['ovr', 'multinomial'],
    #                           'solver': ['sag', 'lbfgs', 'newton-cg'],
    #                           'warm_start': [True],
    #                           'C': C,
    #                           'class_weight': ['balanced'],
    #                           'max_iter': [10000],
    #                           }
    # clf_params.append([LogisticRegression(), logr_hyper_param_space, glob.clfType_list[4]])
    #
    # # KNeighbors Classification
    # knc_hyper_param_space = {'n_neighbors': [3, 5, 7],
    #                          'weights': ['uniform', 'distance'],
    #                          'algorithm': ['auto'],
    #                          'p': [2]}
    # clf_params.append([KNeighborsClassifier(), knc_hyper_param_space, glob.clfType_list[5]])
    #
    # # Gaussian Processes Classification
    # gpc_hyper_param_space = {'kernel': [None],
    #                          'n_restarts_optimizer': [0, 1, 2],
    #                          'multi_class': ['one_vs_rest', 'one_vs_one'],
    #                          'warm_start': [True]}
    # clf_params.append([GaussianProcessClassifier(), gpc_hyper_param_space, glob.clfType_list[6]])
    #
    # # Gaussian Naive Bayes Classification
    # gnb_hyper_param_space = {#'priors': [0.12, 0.75, 0.13],
    #                          }
    # clf_params.append([GaussianNB(), gnb_hyper_param_space, glob.clfType_list[7]])
    #
    # # Linear Discriminant Analysis
    # lda_hyper_param_space = {'solver': ['svd'],
    #                          #'priors': np.array([0.10, 0.15, 0.60, 0.15]),
    #                          'store_covariance': [True, False],
    #                          'tol': [1e-3, 1e-4]}
    # clf_params.append([LinearDiscriminantAnalysis(), lda_hyper_param_space, glob.clfType_list[8]])

    # # Quadratic Discriminant Analysis
    # qda_hyper_param_space = {'priors': np.array([0.10, 0.15, 0.60, 0.15]),
    #                          'reg_param': [0.0, 0.1]}
    # clf_params.append([QuadraticDiscriminantAnalysis(), qda_hyper_param_space, glob.clfType_list[9]])

    # Gradient Boosting Classifier (Tree Based)
    # gbc_hyper_param_space = {
    #                          'loss': ['deviance'],
    #                          "learning_rate": [0.025, 0.05, 0.1, 0.15],
    #                          #"min_samples_split": np.linspace(0.1, 0.5, 12),
    #                          #"min_samples_leaf": np.linspace(0.1, 0.5, 12),
    #                          "max_depth": [3, 5, 7],
    #                          "max_features": ["log2", "sqrt"],
    #                          "criterion": ["friedman_mse"],
    #                          "subsample": [0.5, 0.75, 1.0],
    #                          'n_estimators': np.arange(100, 501, 100)
    #                         }
    # clf_params.append([GradientBoostingClassifier(), gbc_hyper_param_space, glob.clfType_list[10]])

    # XG Boost Classifier
    xgbc_hyper_param_space = xgb_hyper_param_space()
    clf_params.append([xgb.XGBClassifier(), xgbc_hyper_param_space, gbl.clfType_list[11]])

    for clf_p in clf_params:
        print("%s: Running RandomizedSearchCV with %s: %d OF %d FEATS" % (task_name, clf_p[2], num_feats, num_feats_total))
        if clf_p[2] in ['gbc', 'xgbc']:
            scoring = None
        else:
            scoring = performance_metric
        clf_all.append([
                        RandomizedSearchCV(clf_p[0], param_distributions=clf_p[1], n_iter=n_iter, n_jobs=-1, scoring=scoring,
                                           cv=cv_folds, verbose=1)
                       .fit(feat_frame_train, y_train.iloc[:, 0]),
                        clf_p[2],
                        normIdx_train,
                        num_feats,
                        t_feats_idx_subset
                        ])
    # print("CLASSIFY() TOOK %.2f SEC" % (time.time()-t0))
    return clf_all
