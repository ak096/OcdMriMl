from sklearn.metrics import mean_absolute_error
import pandas as pd


#  best_model =  {'GridObject': , 'est_type': , 'normType_train': , 'num_feats': },
def predict_report(feat_set_est_class, best_model, pat_frame_test, feats, pat_y_test):

    print("Best trained model for %s is %s on normed TrainingSet type %s with %d number of features" %
          (feat_set_est_class, best_model['est_type'], best_model['normType_train'], best_model['num_feats']))
    predictions = best_model['GridObject'].predict(pat_frame_test.loc[:, feats])
    pred_score = mean_absolute_error(pat_y_test, predictions)
    print("%s: %.3f" % (best_model['GridObject'].scorer_, pred_score))

    return pd.DataFrame(index=pat_frame_test.index.values.tolist() + ['pred_score'],
                        columns=[feat_set_est_class + '_' + best_model['est_type']],
                        data=predictions.tolist() + [pred_score])


def write_report(writer, best_models_results, pat_y_test_clr, pat_y_test_reg):
    reg_frames = []
    clr_frames = []
    feat_data = {}
    param_data = {}
    for key, value in best_models_results.items():

        # gather prediction results
        est_class = value['est_class']
        pred_results = value['pred_results']
        if est_class == 'reg':
            reg_frames.append(pred_results)
        elif est_class == 'clr':
            clr_frames.append(pred_results)

        est_type = value['best_model']['est_type']

        # gather features
        feat_data[key + '_' + est_type] = pd.Series(value['features'])

        # gather estimator parameters
        param_data[key + '_' + est_type] = pd.Series(value['best_model']['GridObject'].best_params_)

    reg_frames.append(pat_y_test_reg)
    clr_frames.append(pat_y_test_clr)

    reg_frame_report = pd.concat(reg_frames, axis=1, sort=False)
    clr_frame_report = pd.concat(clr_frames, axis=1, sort=False)
    reg_frame_report.to_excel(writer, 'reg_results')
    clr_frame_report.to_excel(writer, 'clr_results')

    feat_report = pd.DataFrame(data=feat_data)
    feat_report.to_excel(writer, 'feature_sets')

    param_report = pd.DataFrame(data=param_data)
    param_report.to_excel(writer, 'best_parameters')

    return
