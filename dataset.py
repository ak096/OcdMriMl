import random
from copy import copy

import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler, SVMSMOTE

from scale import scale, test_set_scale
import gbl


class Subs:
    def __init__(self, tgt_name, test_size=0.15):
        self.test_size = test_size
        self.over_sampler = 'None'
        self.resampled = False

        self.pat_frame = gbl.pat_frame.copy()
        self.con_frame = gbl.con_frame.copy()

        self.pat_frame_stats = gbl.pat_frame_stats.copy()

        self.n_rand_feat = copy(gbl.n_rand_feat)

        self.num_pats = self.pat_frame.shape[0]
        self.pat_names = self.pat_frame.index.tolist()
        self.num_cons = self.con_frame.shape[0]

        # read in target vector according to tgt variable
        self.y_tgt = pd.DataFrame({tgt_name: self.pat_frame_stats.loc[:, tgt_name]})
        if 'class' in tgt_name:
            self.tgt_task = gbl.clf
            self.y_strat = self.y_tgt.copy()
        else:
            self.tgt_task = gbl.reg
            y_strat_name = 'YBOCS_class3'
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

        self.pat_names_test, self.pat_names_train = self._split_test_train_names(self.y_strat, self.test_size,
                                                                                 self.num_bins)

        # check if test and train names are mutually exclusive and add up to total observations
        result = any(elem in self.pat_names_train for elem in self.pat_names_test)
        print(result)
        print('%d pat_names_test' % len(self.pat_names_test))
        print('%d pat_names_train' % len(self.pat_names_train))
        print(set(self.pat_names) == set(self.pat_names_test + self.pat_names_train))

        self.cv_folds = 10
        # assign train and test
        self.pat_frame_train = self.pat_frame.loc[self.pat_names_train, :]
        self.pat_frame_train_y = self.y_tgt.loc[self.pat_names_train, :]

        self.pat_frame_test = self.pat_frame.loc[self.pat_names_test, :]
        self.pat_frame_test_y = self.y_tgt.loc[self.pat_names_test, :]

        # save base copies from train for different oversampling
        self._pat_frame_train_base = self.pat_frame_train.copy(deep=True)
        self._pat_frame_train_y_base = self.pat_frame_train_y.copy(deep=True)

        #  scale data
        self.pat_frame_train_norm, self.pat_train_scaler = scale(self.pat_frame_train)
        self.pat_frame_test_norm = test_set_scale(self.pat_frame_test, self.pat_train_scaler)

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
                a, b = o_sampler.fit_resample(self._pat_frame_train_base.values, self._pat_frame_train_y_base.values)

            except():
                print('oversampling failed')
                return

            self.pat_frame_train = pd.DataFrame(columns=self._pat_frame_train_base.columns, data=a)
            self.pat_frame_train_y = pd.DataFrame(columns=self._pat_frame_train_y_base.columns, data=b)

            for k, v in self.pat_names_train_bins.items():
                self.pat_names_train_bins[k].clear()
                self.pat_names_bins[k].clear()

            for idx, value in enumerate(self.pat_frame_train_y.iloc[:, 0].values):
                self.pat_names_train_bins[value].append(self.pat_frame_train_y.index[idx])

            self.pat_frame_train_norm, self.pat_train_scaler = scale(self.pat_frame_train)
            self.pat_frame_test_norm = test_set_scale(self.pat_frame_test, self.pat_train_scaler)
            self.resampled = True
            print("pat_frame_train over-sampled by %s from %d to %d" % (self.over_sampler,
                                                                        self.num_pats-len(self.pat_names_test),
                                                                        self.pat_frame_train.shape[0]))

    def _split_test_train_names(self, y_strat, test_size, num_bins):
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
            if select_num <= 3:
                select_num = 3
            self.pat_names_test_bins[key] = random.sample(value, select_num)
            self.pat_names_train_bins[key] = [name for name in value if name not in self.pat_names_test_bins[key]]
        pat_names_test = sum(self.pat_names_test_bins.values(), [])
        pat_names_train = sum(self.pat_names_train_bins.values(), [])
        return pat_names_test, pat_names_train


    def check_nan_finite(self):
        pass