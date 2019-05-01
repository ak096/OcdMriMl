from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge, LassoLars
import glob
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, \
                             AdaBoostClassifier, \
                             GradientBoostingClassifier, \
                             GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.gaussian_process.kernels import RBF
import numpy as np
from numpy.random import uniform, randint
import time
import xgboost as xg


C = np.arange(1, 1001, 100)


def svm_hyper_param_space(est_type):
    svm_hps = {'kernel': ['linear', 'rbf'],
               'C': C,
               'gamma': ['scale', 'auto']}

    if est_type == 'reg':
        svm_hps['epsilon'] = [0.3, 0.5, 0.7, 0.9]

    return svm_hps


def mlp_hyper_param_space(num_samples, est_type):

    mlp_hps = {'hidden_layer_sizes': [[int(num_samples*1.5), int(num_samples*0.7)],
                                      #[int(num_samples*2), int(num_samples)],
                                      [int(num_samples*2.5), int(num_samples*1.3)]],
               'activation': ['logistic', 'relu'],
               #'learning_rate': ['constant'],
               'solver': ['adam'],
               #'momentum': [0.6, 0.7, 0.8, 0.9],
               'momentum': [0.9],
               'warm_start': [True],
               'early_stopping': [True],
               #'power_t': [0.3, 0.4, 0.5, 0.6, 0.7],
               #'max_iter': np.arange(150, 301, 50),
               'max_iter': [500],
               #'beta_1': [0.6, 0.7, 0.8, 0.9],
               'beta_1': [0.5],
               #'beta_2': [0.85, 0.89, 0.95, 0.999],
               'batch_size': [4]}

    if est_type == 'reg':
        mlp_hps['activation'].append('tanh')

    return mlp_hps


# return best regressors in a list with order rfr, svmr, mlpr, lr, enr, rr, lasr, laslarr
def regress(feat_frame_train, y_train, cv_folds, performance_metric, normIdx_train, task_name, num_feats_total):

    t0 = time.time()

    num_samples = feat_frame_train.shape[0]
    num_feats = feat_frame_train.shape[1]
    reg_params = []
    reg_all = []

    alpha = np.arange(1, 5)*0.1

    # Random Forest Regression
    rfr_hyper_param_space = {'n_estimators': np.arange(50, 201, 50)}
    reg_params.append([RandomForestRegressor(), rfr_hyper_param_space, glob.regType_list[0]])


    # SVM Regression
    svmr_hyper_param_space = svm_hyper_param_space('reg')
    reg_params.append([SVR(cache_size=2000, max_iter=10000), svmr_hyper_param_space, glob.regType_list[1]])


    # MLP Regression
    # mlpr_hyper_param_space = mlp_hyper_param_space(num_samples, 'reg')
    # reg_params.append([MLPRegressor(), mlpr_hyper_param_space, glob.regType_list[2]])


    # Linear Regression
    lr_hyper_param_space = {}
    reg_params.append([LinearRegression(), lr_hyper_param_space, glob.regType_list[3]])


    # ElasticNet Regression
    # enr_hyper_param_space = {'l1_ratio': np.arange(0, 11)*0.1,
    #                          'max_iter': [2000],
    #                          'precompute': ['auto']}
    # reg_params.append([ElasticNet(), enr_hyper_param_space, glob.regType_list[4]])

    # Ridge Regression
    rr_hyper_param_space = {'alpha': alpha}
    reg_params.append([Ridge(), rr_hyper_param_space, glob.regType_list[5]])


    # Lasso Regression
    lasr_hyper_param_space = {'alpha': alpha}
    reg_params.append([Lasso(), lasr_hyper_param_space, glob.regType_list[6]])


    # LassoLARS Regression
    laslarr_hyper_param_space = {'alpha': alpha}
    reg_params.append([LassoLars(), laslarr_hyper_param_space, glob.regType_list[7]])

    # Gradient Boosting Regressor (Tree Based)
    gbr_hyper_param_space = {'loss': ['ls','lad','huber'],
                             "learning_rate": [0.025, 0.05, 0.075, 0.1, 0.125, 0.15],
                             "min_samples_split": np.linspace(0.1, 0.5, 12),
                             "min_samples_leaf": np.linspace(0.1, 0.5, 12),
                             "max_depth": [3, 5],
                             "max_features": ["log2", "sqrt"],
                             "criterion": ["friedman_mse", "mae"],
                             "subsample": [0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
                             'n_estimators': np.arange(100, 501, 50)}
    reg_params.append([GradientBoostingRegressor(), gbr_hyper_param_space, glob.regType_list[8]])

    # XG Boost Regressor
    xgbr_hyper_param_space = {"colsample_bytree": [uniform(0.7, 0.3)],
                              "gamma": [uniform(0, 0.5)],
                              "learning_rate": [uniform(0.03, 0.3)], # default 0.1
                              "max_depth": [randint(2, 6)], # default 3
                              "n_estimators": np.arange(100, 501, 50), # default 100
                              "subsample": [uniform(0.6, 0.4)]}
    reg_params.append([xg.XGBRegressor(), xgbr_hyper_param_space, glob.regType_list[9]])

    for reg_p in reg_params:
        print("%s: Running RandomizedSearchCV with %s: %d OF %d FEATS" % (task_name, reg_p[2], num_feats, num_feats_total))
        if reg_p[2] in ['gbr', 'xgbr']:
            scoring = None
        else:
            scoring = performance_metric
        reg_all.append([RandomizedSearchCV(reg_p[0], param_distributions=reg_p[1], n_jobs=-1, scoring=scoring,
                                     cv=cv_folds, verbose=0, iid=True).fit(feat_frame_train, y_train.iloc[:, 0]),
                        reg_p[2],
                        normIdx_train,
                        num_feats])
    # print("REGRESS() TOOK %.2f SEC" % (time.time()-t0))
    return reg_all


# return best classifiers in a list with order rfc, svmc, mlpc, abc, logr, knc, gpc, gnb, lda, qda
def classify(feat_frame_train, y_train, cv_folds, performance_metric, normIdx_train, task_name, num_feats_total):

    t0 = time.time()

    num_samples = feat_frame_train.shape[0]
    num_feats = feat_frame_train.shape[1]
    clf_params = []
    clf_all = []

    # Random Forest Classification
    rfc_hyper_param_space = {'n_estimators': np.arange(50, 201, 50),
                             'warm_start': [True]}
    clf_params.append([RandomForestClassifier(), rfc_hyper_param_space, glob.clfType_list[0]])

    # SVM Classification
    svmc_hyper_param_space = svm_hyper_param_space('clf')
    clf_params.append([SVC(cache_size=2000, max_iter=10000), svmc_hyper_param_space, glob.clfType_list[1]])

    # MLP Classification
    # mlpc_hyper_param_space = mlp_hyper_param_space(num_samples, 'clf')
    # clf_params.append([MLPClassifier(), mlpc_hyper_param_space, glob.clfType_list[2]])

    # Adaboost Classification
    # abc_hyper_param_space = {'n_estimators': np.arange(50, 101, 10),
    #                         'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.15, 0.2, 0.25, 0.3]}
    # clf_params.append([AdaBoostClassifier(), abc_hyper_param_space, glob.clfType_list[3]])

    # Logistic Regression
    logr_hyper_param_space = {'penalty': ['l2'],
                              'multi_class': ['ovr', 'multinomial'],
                              'solver': ['newton-cg', 'sag', 'lbfgs'],
                              'warm_start': [True],
                              'C': C,
                              'class_weight': ['balanced'],
                              'max_iter': [10000]}
    # if feat_frame_train.shape[0] <= feat_frame_train.shape[1]:
    #     logr_hyper_param_space['dual'] = [True]
    clf_params.append([LogisticRegression(), logr_hyper_param_space, glob.clfType_list[4]])

    # KNeighbors Classification
    knc_hyper_param_space = {'n_neighbors': [3, 5, 7],
                             'weights': ['uniform', 'distance'],
                             'algorithm': ['auto'],
                             'p': [2]}
    clf_params.append([KNeighborsClassifier(), knc_hyper_param_space, glob.clfType_list[5]])

    # Gaussian Processes Classification
    gpc_hyper_param_space = {'kernel': [None],
                             'n_restarts_optimizer': [0, 1, 2],
                             'multi_class': ['one_vs_rest'],
                             'warm_start': [True]}
    clf_params.append([GaussianProcessClassifier(), gpc_hyper_param_space, glob.clfType_list[6]])

    # Gaussian Naive Bayes Classification
    gnb_hyper_param_space = {#'priors': [0.12, 0.75, 0.13],
                             }
    clf_params.append([GaussianNB(), gnb_hyper_param_space, glob.clfType_list[7]])

    # Linear Discriminant Analysis
    lda_hyper_param_space = {'solver': ['svd']
                             #'shrinkage': ['auto'],
                             #'priors': np.array([0.10, 0.15, 0.60, 0.15]),
                            }
    clf_params.append([LinearDiscriminantAnalysis(), lda_hyper_param_space, glob.clfType_list[8]])

    # # Quadratic Discriminant Analysis
    # qda_hyper_param_space = {'priors': np.array([0.10, 0.15, 0.60, 0.15]),
    #                          'reg_param': [0.0, 0.1]}
    # clf_params.append([QuadraticDiscriminantAnalysis(), qda_hyper_param_space, glob.clfType_list[9]])

    # Gradient Boosting Classifier (Tree Based)
    gbc_hyper_param_space = {'loss': ['deviance'],
                             "learning_rate": [0.025, 0.05, 0.075, 0.1, 0.125, 0.15],
                             "min_samples_split": np.linspace(0.1, 0.5, 12),
                             "min_samples_leaf": np.linspace(0.1, 0.5, 12),
                             "max_depth": [3, 5],
                             "max_features": ["log2", "sqrt"],
                             "criterion": ["friedman_mse", "mae"],
                             "subsample": [0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
                             'n_estimators': np.arange(100, 501, 50)}
    clf_params.append([GradientBoostingClassifier(), gbc_hyper_param_space, glob.clfType_list[10]])

    # XG Boost Classifier
    xgbc_hyper_param_space = {"colsample_bytree": [uniform(0.7, 0.3)],
                              "gamma": [uniform(0, 0.5)],
                              "learning_rate": [uniform(0.03, 0.3)], # default 0.1
                              "max_depth": [randint(2, 6)], # default 3
                              "n_estimators": np.arange(100, 501, 50), # default 100
                              "subsample": [uniform(0.6, 0.4)]}
    clf_params.append([xg.XGBClassifier(), xgbc_hyper_param_space, glob.clfType_list[11]])

    for clf_p in clf_params:
        print("%s: Running RandomizedSearchCV with %s: %d OF %d FEATS" % (task_name, clf_p[2], num_feats, num_feats_total))
        if clf_p[2] in ['gbc', 'xgbc']:
            scoring = None
        else:
            scoring = performance_metric
        clf_all.append([RandomizedSearchCV(clf_p[0], param_distributions=clf_p[1], n_jobs=-1, scoring=scoring,
                                     cv=cv_folds, verbose=0, iid=True).fit(feat_frame_train, y_train.iloc[:, 0]),
                        clf_p[2],
                        normIdx_train,
                        num_feats])
    # print("CLASSIFY() TOOK %.2f SEC" % (time.time()-t0))
    return clf_all
