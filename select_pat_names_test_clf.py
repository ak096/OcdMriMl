import random


def select_pat_names_test_clf(y_clf, clf_tgt, test_size, num_classes):
    random.seed(random.randint(1, 101))
    target_list = y_clf.loc[:, clf_tgt].values.tolist()
    pat_names_classes_keys = list(set(target_list))

    pat_names_classes = {key: [] for key in pat_names_classes_keys}

    for idx, value in enumerate(target_list):
        # print('adding %s at idx %d to class %d' % (y_clf.index[idx], idx, value))
        pat_names_classes[value].append(y_clf.index[idx])
    # print(pat_names_classes)
    # count = 0
    pat_names_test = []
    for key, value in pat_names_classes.items():
        pat_names_class_num = len(value)
        if num_classes > 2:
            select_num = 5
        else:
            select_num = round(test_size*pat_names_class_num)
        pat_names_class_rand_idx = random.sample(range(pat_names_class_num), select_num)
        pat_names_test += [value[i] for i in pat_names_class_rand_idx]

    return pat_names_test
