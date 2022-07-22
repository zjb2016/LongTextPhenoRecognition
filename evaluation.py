#!/usr/bin/python3
import logging

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, f1_score
from torch import tensor
import seaborn as sns
from tqdm import tqdm


def exmp_accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_true, y_pred = y_true.astype('bool').astype('int'), y_pred.astype('bool').astype('int')
    return ((y_true * y_pred).sum(axis=1) / (
            np.finfo(float).eps + (y_true + y_pred).astype('bool').astype('int8').sum(axis=1))).mean()


def multi_label_metrics(logits, labels, threshold=0.5):
    logits = tensor(logits)
    y_true = tensor(labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits)
    y_pred = np.zeros(probs.shape)
    # if np.array(probs).max() < threshold:
    #     threshold = np.array(probs).mean()
    y_pred[np.where(probs >= threshold)] = 1

    # compute metrics
    hamming_score = exmp_accuracy(y_true, y_pred)
    precision, recall, f_score, support = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    subset_acc = accuracy_score(y_true, y_pred)

    logging.info('==== print every batch eval info =====\n')
    logging.info('==== output logits =====\n')
    logging.info(logits)
    logging.info('==== true labels =====\n')
    logging.info(labels)
    logging.info('==== y_preds =====\n')
    logging.info(y_pred)

    logging.info('==== evaluation =====')
    logging.info('support: %s' % str(support))
    logging.info('subset acc: %s' % str(subset_acc))
    logging.info('hamming_score: %s' % str(hamming_score))
    logging.info('precision: %s' % str(precision))
    logging.info('recall: %s' % str(recall))
    logging.info('f_score: %s' % str(f_score))

    metrics = {'f1': f_score,
               's_accuracy': subset_acc,
               'hamming_score': hamming_score,
               'recall': recall,
               'precision': precision,
               'support': support
               }
    return metrics


def summary_train(training_stats):
    # Display floats with two decimal places.
    pd.set_option('display.precision', 3)
    # Create a DataFrame from our training statistics.
    df_stats = pd.DataFrame(data=training_stats)
    # Use the 'epoch' as the row index.
    df_stats = df_stats.set_index('epoch')
    df_stats.to_csv('training_summary.csv')
    return df_stats


def draw_train_stats(df_stats, epochs):
    # Use plot styling from seaborn.
    sns.set_style('darkgrid')
    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12, 6)
    # Plot the learning curve.
    plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
    plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")
    plt.plot(df_stats['Valid. F1'], 'r-o', label="F1")
    # Label the plot.
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    arr = [i + 1 for i in range(epochs)]
    plt.xticks(arr)
    plt.savefig('training_loss.jpg')
    # plt.show()


class F1Score:
    '''
    F1 Score
    binary:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
    micro:
            Calculate metrics globally by considering each element of the label
            indicator matrix as a label.
    macro:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
    weighted:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label).
    samples:
            Calculate metrics for each instance, and find their average.
    Example:
        >>> metric = F1Score(**)
        >>> for epoch in range(epochs):
        >>>     metric.reset()
        >>>     for batch in batchs:
        >>>         logits = model()
        >>>         metric(logits,target)
        >>>         print(metric.name(),metric.value())
    '''

    def __init__(self, thresh=0.5, normalizate=True, task_type='binary', average='weighted', search_thresh=False):
        super(F1Score).__init__()
        self.y_true = 0
        self.y_pred = 0
        assert task_type in ['binary', 'multiclass']
        assert average in ['binary', 'micro', 'macro', 'samples', 'weighted']

        self.thresh = thresh
        self.task_type = task_type
        self.normalizate = normalizate
        self.search_thresh = search_thresh
        self.average = average

    def thresh_search(self, y_prob):
        '''
        对于f1评分的指标，一般我们需要对阈值进行调整，一般不会使用默认的0.5值，因此
        这里我们对Thresh进行优化
        :return:
        '''
        best_threshold = 0
        best_score = 0
        for threshold in tqdm([i * 0.01 for i in range(100)], disable=True):
            self.y_pred = y_prob > threshold
            score = self.value()
            if score > best_score:
                best_threshold = threshold
                best_score = score
        return best_threshold, best_score

    def __call__(self, logits, target):
        '''
        计算整个结果
        :return:
        '''
        self.y_true = target.cpu().numpy()
        if self.normalizate and self.task_type == 'binary':
            y_prob = logits.sigmoid().data.cpu().numpy()
        elif self.normalizate and self.task_type == 'multiclass':
            y_prob = logits.softmax(-1).data.cpu().detach().numpy()
        else:
            y_prob = logits.cpu().detach().numpy()

        if self.task_type == 'binary':
            if self.thresh and self.search_thresh == False:
                self.y_pred = (y_prob > self.thresh).astype(int)
                self.value()
            else:
                thresh, f1 = self.thresh_search(y_prob=y_prob)
                self.thresh = thresh
                print(f"Best thresh: {thresh:.4f} - F1 Score: {f1:.4f}")

        if self.task_type == 'multiclass':
            self.y_pred = np.argmax(y_prob, 1)

    def get_thresh(self):
        return self.thresh

    def reset(self):
        self.y_pred = 0
        self.y_true = 0

    def value(self):
        f1 = f1_score(y_true=self.y_true, y_pred=self.y_pred, average=self.average)
        return f1

    def name(self):
        return 'f1'
