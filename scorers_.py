class RegScorer:
    def __init__(self):
        self.explained_variance = 'explained_variance'
        self.max_error = 'max_error'

        self.neg_mean_absolute_error = 'neg_mean_absolute_error'
        self.neg_median_absolute_error = 'neg_median_absolute_error'

        self.neg_mean_square_error = 'neg_mean_squared_error'
        self.neg_mean_squared_log_error = 'neg_mean_squared_log_error'

        self.r2 = 'r2'


class ClfScorer:
    def __init__(self):
        self.accuracy = 'accuracy',
        self.balanced_accuracy = 'balanced_accuracy'

        self.brier_score_loss = 'brier_score_loss'
        self.neg_log_loss = 'neg_log_loss' #requires predict_proba support
        self.roc_auc = 'roc_auc'

        self.f1 =  'f1' #for binary targets
        self.f1_micro = 'f1_micro' #micro averaged
        self.f1_macro = 'f1_macro',#macro-averaged
        self.f1_weighted = 'f1_weighted' #weighted average
        # self.f1_samples =  'f1_samples' #by multilabel sample

        self.precision =  'precision'
        self.precision_micro = 'precision_micro'
        self.precision_macro = 'precision_macro'
        self.precision_weighted = 'precision_weighted'
        # self.precision_samples =

        self.recall = 'recall'
        self.recall_micro = 'recall_micro'
        self.recall_macro = 'recall_macro'
        self.recall_weighted = 'recall_weighted'
        # self.recall_samples =

        self.jaccard = 'jaccard'
        self.jaccard_micro = 'jaccard_micro'
        self.jaccard_macro = 'jaccard_macro'
        self.jaccard_weighted = 'jaccard_weighted'
        # self.jaccard_samples =