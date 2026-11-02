import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Adapted from Decision Score Histogram by Emmanuel Aboah Boateng (2022).
# Original: https://github.com/emmanuelaboah/Decision-Score-Histogram
# MIT License: see LICENSE_Decision-Score-Histogram.txt
from sklearn.preprocessing import MinMaxScaler as _MinMaxScaler

class MinMaxScaler(_MinMaxScaler):
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.size == 0:
            X = np.zeros((1, 1), dtype=float)
        else:
            X = X.reshape(-1, 1)
        return super().fit(X, y)

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if X.size == 0:
            return np.array([])
        return super().transform(X.reshape(-1, 1)).ravel()


def save_plot(fig, path="plots", tight_layout=True, fig_extension="png", resolution=400):
    """
        Function for saving figures and plots
        :arg
            1. fig: label of the figure
            2. path (optional): output path of the figure
    """
    img_path = os.path.join(".", path)
    os.makedirs(img_path, exist_ok=True)
    fig_path = os.path.join(img_path, fig + "." + fig_extension)

    print("Saving figure...", fig)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(fig_path, format=fig_extension, dpi=resolution)
    print("figure can be found in: ", path)


class AnomalyScoreHist:
    """
        Histogram based visualization approach for
        anomaly detection algorithms and supervised classification
        algorithms involving binary predictions.
    """

    def __init__(self, decision_scores, ground_truth):
        self.decision_scores = decision_scores
        self.ground_truth = ground_truth

    def check_input_dtype(self, input_data):
        if isinstance(input_data, np.ndarray):
            return input_data.reshape(-1, 1)

        elif isinstance(input_data, (pd.DataFrame, pd.Series)):
            return input_data.to_numpy().reshape(-1, 1)

        else:
            try:
                data = np.array(input_data)
            except:
                raise Exception('Wrong data type')
            else:
                return data.reshape(-1, 1)

    def check_gtruth_dtype(self, ground_truth):
        g_truth = self.check_input_dtype(ground_truth)
        warning_msg = 'ground truth must be an array of 1s (positive scores) and -1s (negative scores)'
        assert all(label in [1, -1] for label in g_truth), warning_msg

        return g_truth

    def compute_hist_data(self, dec_score, ground_truth):
        dec_score = self.check_input_dtype(self.decision_scores)
        ground_truth = self.check_gtruth_dtype(self.ground_truth)

        false_positives = []
        false_negatives = []
        for idx in range(len(dec_score)):
            if ground_truth[idx] == -1 and dec_score[idx] >= 0:
                false_positives.append(dec_score[idx])

            elif ground_truth[idx] == 1 and dec_score[idx] < 0:
                false_negatives.append(dec_score[idx])

            else:
                continue

        true_positives = []
        true_negatives = []
        for idx in range(len(dec_score)):
            if ground_truth[idx] == 1 and dec_score[idx] >= 0:
                true_positives.append(dec_score[idx])

            elif ground_truth[idx] == -1 and dec_score[idx] < 0:
                true_negatives.append(dec_score[idx])

            else:
                continue

        # Handle situations where input data is binary instead of scores (real numbers)
        if len(np.unique(true_positives)) == 1 and true_positives[0] > 0:
            if np.unique(false_positives) == np.unique(true_positives) or len(false_positives) == 0:

                # Means input is binary
                true_positives = [1] * len(true_positives)
                false_positives = [1] * len(false_positives)

        else:
            # scores (real numbers) were provided
            all_positives = [dec_score[idx] for idx in range(len(dec_score))
                         if dec_score[idx] >= 0]
            pos_scaler = MinMaxScaler().fit(all_positives)

            if len(true_positives) == 0:
                true_positives = []
            else:
                true_positives = np.round(pos_scaler.transform(true_positives), 3)

            if len(false_positives) == 0:
                false_positives = []
            else:
                false_positives = np.round(pos_scaler.transform(false_positives), 3)

        all_negatives = [dec_score[idx] for idx in range(len(dec_score))
                         if dec_score[idx] < 0]
        neg_scaler = MinMaxScaler().fit(all_negatives)

        if len(true_negatives) == 0:
            true_negatives = []
        else:
            true_negatives = np.round(neg_scaler.transform(true_negatives) * -1, 3)

        if len(false_negatives) == 0:
            false_negatives = []
        else:
            false_negatives = np.round(neg_scaler.transform(false_negatives) * -1, 3)

        return true_positives, true_negatives, false_positives, false_negatives

    def flip_negative_plot(self, negative_scores_norm):
        '''
        Because of the normalization of negative scores
        the negative half of the histogram needs to be flipped
        which is what this function does.
        '''
        remainder, quotient = np.modf(-2 - negative_scores_norm)
        final_neg_scores = np.where(quotient == -2, -1, remainder)

        return final_neg_scores

    def plot_hist(self, fig_name='hist_plot'):
        TP, TN, FP, FN = self.compute_hist_data(
                                self.decision_scores, self.ground_truth)

        if len(TN) != 0:
            TN = self.flip_negative_plot(TN)

        if len(FN) != 0:
            FN = self.flip_negative_plot(FN)

        plt.figure(figsize=(9, 7))
        plt.hist(TP, weights=np.ones(len(TP)) / (len(TP) + len(FN)), facecolor='C0',
                 bins=50, label='True Positive', alpha=0.5)
        plt.hist(TN, weights=np.ones(len(TN)) / (len(TN) + len(FP)), facecolor='C1',
                 bins=50, label='True Negative', alpha=0.5)
        plt.hist(FP, weights=np.ones(len(FP)) / (len(TN) + len(FP)), facecolor='C1',
                 bins=50, label='False Positive')
        plt.hist(FN, weights=np.ones(len(FN)) / (len(TP) + len(FN)), facecolor='C0',
                 bins=50, label='False Negative')

        plt.xlabel('Decision Score', fontsize=27, labelpad=10)
        plt.ylabel('Prediction', fontsize=27, labelpad=10)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlim([-1.1, 1.1])
        plt.ylim([0, 1.05])
        plt.locator_params(axis='x', nbins=5)
        plt.locator_params(axis='y', nbins=5)
        plt.axvline(x=0, linestyle='--', linewidth=1, color='black')
        plt.legend(loc='best', fontsize=18)
        save_plot(fig_name)

        plt.show()
