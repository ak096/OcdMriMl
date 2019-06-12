import os
from fs_read import fs_data_collect
import pandas as pd
import numpy as np
from gdrive import get_pat_stats
from sklearn.feature_selection import VarianceThreshold
import gbl
from imblearn.over_sampling import RandomOverSampler, SVMSMOTE
import random
from scale import scale, test_set_scale
from numpy.random import randint
import random
from math import floor


class Subs:
    def __init__(self, tgt_name, test_size=0.15):
        self.test_size = test_size
        self.over_sampler = 'None'
        self.resampled = False
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
        self.pat_frame = pd.concat([self.pat_frame, self.pat_frame_stats.loc[:, gbl.clin_demog_feats]], axis=1,
                                   sort=False)

        # read in target vector according to tgt variable
        self.y_tgt = pd.DataFrame({tgt_name: self.pat_frame_stats.loc[:, tgt_name]})
        if 'class' in tgt_name:
            self.tgt_task = 'clf'
            self.y_strat = self.y_tgt
        else:
            self.tgt_task = 'reg'

            y_strat_name = 'YBOCS_class3_scorerange'
            self.y_strat = pd.DataFrame({y_strat_name: self.pat_frame_stats.loc[:, y_strat_name]})

        # extract train and test set names

        # pat_names_train, pat_names_test = train_test_split(pat_names,
        #                                                    test_size=self.test_size,
        #                                                    stratify=y_clf)
        #                                                    #random_state=random.randint(1, 101))
        self.pat_names_bins = {}
        self.pat_names_test_bins = {}
        self.pat_names_train_bins = {}
        self.bin_keys = np.unique(self.y_strat.iloc[:, 0])
        self.num_bins = len(self.bin_keys)
        self.multiclass = False
        if self.tgt_task is 'clf' and self.num_bins > 2:
            self.multiclass = True

        self.pat_names_test, self.pat_names_train = self.split_test_train_names(self.y_strat, self.test_size,
                                                                                self.num_bins)

        # check if test and train names are mutually exclusive and add up to total observations
        result = any(elem in self.pat_names_train for elem in self.pat_names_test)
        print(result)
        print('%d pat_names_test' % len(self.pat_names_test))
        print('%d pat_names_train' % len(self.pat_names_train))
        print(set(self.pat_names) == set(self.pat_names_test + self.pat_names_train))

        self.cv_folds = 10

        self.pat_frame_train = self.pat_frame.loc[self.pat_names_train, :]
        self.pat_frame_train_norms, self.pat_train_scalers = scale(self.pat_frame_train)
        self.pat_frame_train_y = self.y_tgt.loc[self.pat_names_train, :]

        self.pat_frame_test = self.pat_frame.loc[self.pat_names_test, :]
        self.pat_frame_test_norms = test_set_scale(self.pat_frame_test, self.pat_train_scalers)
        self.pat_frame_test_y = self.y_tgt.loc[self.pat_names_test, :]

        # if self.tgt_task == 'reg' leave it else
        self.imbalanced_classes = False
        if self.tgt_task == 'clf':
            # check if any classes have more than 1 std away from mean number of observations per class
            if any(abs(self.y_tgt.iloc[:, 0].value_counts() - np.mean(self.y_tgt.iloc[:, 0].value_counts())) >
                   np.std(self.y_tgt.iloc[:, 0].value_counts())):
                self.imbalanced_classes = True

        # con
        self.con_frame_norms, self.con_scalers = scale(self.con_frame)

    def resample(self, over_sampler='None'):
        self.over_sampler = over_sampler
        if self.tgt_task is not 'clf':
            print('cannot resample for non-classification task')
            return
        elif not self.imbalanced_classes:
            print('classes not imbalanced, not resampling')
            self.over_sampler = 'None'
            return
        else:
            if self.over_sampler == 'None':
                print('not resampling, None given')
                return
            elif self.over_sampler == 'ROS':
                o_sampler = RandomOverSampler(random_state=random.randint(1, 101))
            elif self.over_sampler == 'SVMSMOTE':
                o_sampler = SVMSMOTE(random_state=random.randint(1, 101))
            else:
                print('over_sampler not supported')
                return

            try:
                a, b = o_sampler.fit_resample(self.pat_frame_train.values, self.pat_frame_train_y.values)

            except():
                print('oversampling failed')
                return

            self.pat_frame_train = pd.DataFrame(columns=self.pat_frame_train.columns, data=a)
            self.pat_frame_train_y = pd.DataFrame(columns=self.pat_frame_train_y.columns, data=b.astype(int))

            for k, v in self.pat_names_train_bins.items():
                self.pat_names_train_bins[k].clear()
                self.pat_names_bins[k].clear()

            for idx, value in enumerate(self.pat_frame_train_y.iloc[:, 0].values):
                self.pat_names_train_bins[value].append(self.pat_frame_train_y.index[idx])

            self.pat_frame_train_norms, self.pat_train_scalers = scale(self.pat_frame_train)
            self.pat_frame_test_norms = test_set_scale(self.pat_frame_test, self.pat_train_scalers)
            self.resampled = True
            print("pat_frame_train over-sampled by %s from %d to %d" % (self.over_sampler,
                                                                        self.num_pats-len(self.pat_names_test),
                                                                        self.pat_frame_train.shape[0]))

    def split_test_train_names(self, y_strat, test_size, num_bins):
        random.seed(random.randint(1, 101), version=2)
        target_list = y_strat.iloc[:, 0].values.tolist()
        pat_names_bins_keys = list(set(target_list))
        self.pat_names_bins = {key: [] for key in pat_names_bins_keys}
        self.pat_names_test_bins = {key: [] for key in pat_names_bins_keys}
        self.pat_names_train_bins = {key: [] for key in pat_names_bins_keys}

        for idx, value in enumerate(target_list):
            self.pat_names_bins[value].append(y_strat.index[idx])

        for key, value in self.pat_names_bins.items():
            pat_names_bins_num = len(value)
            select_num = round(test_size*pat_names_bins_num)

            self.pat_names_test_bins[key] = random.sample(value, select_num)
            self.pat_names_train_bins[key] = [name for name in value if name not in self.pat_names_test_bins[key]]
        pat_names_test = sum(self.pat_names_test_bins.values(), [])
        pat_names_train = sum(self.pat_names_train_bins.values(), [])
        return pat_names_test, pat_names_train




