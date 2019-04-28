from sklearn.metrics import mean_absolute_error, accuracy_score, balanced_accuracy_score
import pandas as pd
import glob


#  best_model =  {'GridObject': , 'est_type': , 'normIdx_train': , 'num_feats': },
def predict_report(feat_set_est_class, best_model, pat_frame_test, feats, pat_test_y, ec):

    print("Best trained model for %s is %s on normed TrainingSet type %s with %d number of features" %
          (feat_set_est_class, best_model['est_type'], glob.normType_list[best_model['normIdx_train']],
           best_model['num_feats']))
    predictions = best_model['GridObject'].predict(pat_frame_test.loc[:, feats])

    pred_score = best_model['GridObject'].score(pat_frame_test.loc[:, feats], pat_test_y)
    print("%s: %.3f" % (best_model['GridObject'].scorer_, pred_score))
    pred_score2 = -1
    if ec == 'clf':
        pred_score2 = accuracy_score(pat_test_y, predictions)
        print('accuracy_score %.3f' % pred_score2)

    if best_model['est_type'] == 'gbr' or best_model['est_type'] == 'gbc':
        scorer = best_model['GridObject'].best_params_['loss']
    else:
        scorer = best_model['GridObject'].scorer_
    return pd.DataFrame(index=pat_frame_test.index.tolist() + [scorer, 'acc_score'],
                        columns=[feat_set_est_class + '_' + best_model['est_type'] + '_' +
                                 glob.normType_list[best_model['normIdx_train']]],
                        data=predictions.tolist() + [pred_score, pred_score2])


def write_report(writer, best_models_results, pat_test_y_clf, pat_test_y_reg):
    reg_frames = []
    clf_frames = []
    feat_data = {}
    param_data = {}
    for key, value in best_models_results.items():

        # gather prediction results
        est_class = value['est_class']
        pred_results = value['pred_results']
        if est_class == 'reg':
            reg_frames.append(pred_results)
        elif est_class == 'clf':
            clf_frames.append(pred_results)

        est_type = value['best_model']['est_type']
        normType_train = glob.normType_list[value['best_model']['normIdx_train']]

        # gather features
        feat_data[key + '_' + est_type + '_' + normType_train] = pd.Series(value['features'])

        # gather estimator parameters
        param_data[key + '_' + est_type + '_' + normType_train] = \
            pd.Series(value['best_model']['GridObject'].best_params_)

    reg_frames.append(pat_test_y_reg)
    clf_frames.append(pat_test_y_clf)

    reg_frame_report = pd.concat(reg_frames, axis=1, sort=False)
    clf_frame_report = pd.concat(clf_frames, axis=1, sort=False)
    reg_frame_report.to_excel(writer, 'reg_results')
    clf_frame_report.to_excel(writer, 'clf_results')

    feat_report = pd.DataFrame(data=feat_data)
    feat_report.to_excel(writer, 'feature_sets')

    param_report = pd.DataFrame(data=param_data)
    param_report.to_excel(writer, 'best_parameters')

    return
