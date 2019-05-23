import os
from fs_read import fs_data_collect
import pandas as pd
import numpy as np
from gdrive import get_pat_stats
from sklearn.feature_selection import VarianceThreshold
import gbl
from select_pat_names_test_clf import select_pat_names_test_clf
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
import random
from scale import scale, testSet_scale


class Subs:
    def __init__(self, clf_tgt, test_size=0.18, over_sampler='None'):
        self.test_size = test_size
        self.over_sampler = over_sampler
        # get data from FreeSurfer stats
        path_base = os.path.abspath('Desktop/FS_SUBJ_ALL').replace('PycharmProjects/OcdMriMl/', '')

        group = ['con', 'pat']

        self.con_frame = fs_data_collect(group[0], path_base)
        self.pat_frame = fs_data_collect(group[1], path_base)

        pd.DataFrame({'con': self.con_frame.columns[self.con_frame.columns != self.pat_frame.columns],
                      'pat': self.pat_frame.columns[self.con_frame.columns != self.pat_frame.columns]})

        self.num_pats = self.pat_frame.shape[0]
        self.pat_names = self.pat_frame.index.tolist()
        self.num_cons = self.con_frame.shape[0]

        # remove low variance features
        before = self.pat_frame.shape[1]
        threshold = 0.01
        sel = VarianceThreshold(threshold=threshold)
        sel.fit_transform(self.pat_frame)
        retained_mask = sel.get_support(indices=False)
        self.pat_frame = self.pat_frame.loc[:, retained_mask]
        after = self.pat_frame.shape[1]
        print('REMOVING %d FEATS UNDER %.2f VAR: FROM %d TO %d' % (before - after, threshold, before, after))

        # remove features less than 90% populated
        before = self.pat_frame.shape[1]
        threshold = round(1.00*self.pat_frame.shape[0])
        self.pat_frame.dropna(axis='columns', thresh=threshold, inplace=True)
        after = self.pat_frame.shape[1]
        print('REMOVING %d FEATS LESS THAN %.2f PERCENT FILLED: FROM %d TO %d' %
              (before - after, threshold/self.pat_frame.shape[0], before, after))
        gbl.FS_feats = self.pat_frame.columns.tolist()

        # add clin demo features
        self.pat_frame_stats = get_pat_stats()
        if True in self.pat_frame_stats.index != self.pat_frame.index:
            exit("feature and target pats not same")
        self.pat_frame = pd.concat([self.pat_frame, self.pat_frame_stats.loc[:, gbl.demo_clin_feats]], axis=1, sort=False)

        if 'obs' in clf_tgt:
            self.y_reg = pd.DataFrame({'obs': self.pat_frame_stats.loc[:, 'obs']})
        elif 'com' in clf_tgt:
            self.y_reg = pd.DataFrame({'com': self.pat_frame_stats.loc[:, 'com']})
        else:
            self.y_reg = pd.DataFrame({'YBOCS': self.pat_frame_stats.loc[:, 'YBOCS']})

        self.y_clf = pd.DataFrame({clf_tgt: self.pat_frame_stats.loc[:, clf_tgt]})

        # extract train and test set names

        # pat_names_train, pat_names_test = train_test_split(pat_names,
        #                                                    test_size=self.test_size,
        #                                                    stratify=y_clf)
        #                                                    #random_state=random.randint(1, 101))
        self.num_classes = len(np.unique(self.y_clf.iloc[:, 0]))
        self.pat_names_test = select_pat_names_test_clf(self.y_clf, clf_tgt, self.test_size, self.num_classes)
        self.pat_names_train = [name for name in self.pat_names if name not in self.pat_names_test]
        result = any(elem in self.pat_names_train for elem in self.pat_names_test)
        print(result)
        print('%d pat_names_test' % len(self.pat_names_test))
        print('%d pat_names_train' % len(self.pat_names_train))
        print(set(self.pat_names) == set(self.pat_names_test + self.pat_names_train))

        self.pat_frame_train = self.pat_frame.loc[self.pat_names_train, :]

        # create train set for reg-------------------
        self.pat_frame_train_reg = self.pat_frame_train
        self.pat_frame_train_reg_norms, self.pat_train_reg_scalers = scale(self.pat_frame_train_reg)

        self.pat_frame_train_y_reg = self.y_reg.loc[self.pat_names_train, :]

        # create train set for clf-------------------
        self.pat_frame_train_y_clf = self.y_clf.loc[self.pat_names_train, :]

        self.pat_frame_train_clf_list = [0, 1, 2, 3]
        self.pat_frame_train_y_clf_list = [0, 1, 2, 3]

        self.pat_frame_train_clf_list[0] = self.pat_frame_train
        self.pat_frame_train_y_clf_list[0] = self.pat_frame_train_y_clf

        self.over_samp_names = ['None', 'ROS', 'SMOTE', 'ADASYN']

        if any(abs(self.y_clf.iloc[:, 0].value_counts() - np.mean(self.y_clf.iloc[:, 0].value_counts())) >
               np.std(self.y_clf.iloc[:, 0].value_counts())):  # if more than 1 std away from mean than
            self.imbalanced_classes = True
        else:
            self.imbalanced_classes = False
            if self.over_sampler is not 'None':
                print('classes not imbalanced, no oversampling possible')
            self.over_sampler = 'None'

        if self.imbalanced_classes:
            over_samps = [RandomOverSampler(random_state=random.randint(1, 101)),
                          SMOTE(random_state=random.randint(1, 101)),
                          ADASYN(random_state=random.randint(1, 101))]

            for idx, over_samp in enumerate(over_samps):
                try:
                    a, b = over_samp.fit_resample(self.pat_frame_train.values, self.pat_frame_train_y_clf.values)
                    self.pat_frame_train_clf_list[idx + 1] = pd.DataFrame(columns=self.pat_frame_train.columns, data=a)
                    self.pat_frame_train_y_clf_list[idx + 1] = pd.DataFrame(columns=self.pat_frame_train_y_clf.columns, data=b)
                except:
                    pass

        o_s = self.over_samp_names.index(self.over_sampler)
        self.pat_frame_train_clf_norms, self.pat_train_clf_scalers = scale(self.pat_frame_train_clf_list[o_s])
        self.pat_frame_train_y_clf = self.pat_frame_train_y_clf_list[o_s]
        print("pat_fram_train_clf_norm upsampled %s to %d" % (self.over_samp_names[o_s],
                                                              self.pat_frame_train_clf_norms[o_s].shape[0]))

        # create test sets for reg and clr
        self.pat_frame_test = self.pat_frame.loc[self.pat_names_test, :]

        self.pat_frame_test_reg_norms = testSet_scale(self.pat_frame_test, self.pat_train_reg_scalers)
        self.pat_frame_test_y_reg = self.y_reg.loc[self.pat_names_test, :]

        self.pat_frame_test_clf_norms = testSet_scale(self.pat_frame_test, self.pat_train_clf_scalers)
        self.pat_frame_test_y_clf = self.y_clf.loc[self.pat_names_test, :]

        # con
        self.con_frame_norms, self.con_scalers = scale(self.con_frame)

    def set_over_sampled(self, over_sampler='None'):
        if not self.imbalanced_classes:
            print('classes not imbalanced, returning unchanged data set')
            self.over_sampler = 'None'
        o_s = self.over_samp_names.index(self.over_sampler)
        self.pat_frame_train_clf_norms, self.pat_train_clf_scalers = scale(self.pat_frame_train_clf_list[o_s])
        self.pat_frame_train_y_clf = self.pat_frame_train_y_clf_list[o_s]
        self.pat_frame_test_clf_norms = testSet_scale(self.pat_frame_test, self.pat_train_clf_scalers)




