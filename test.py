#!/usr/bin/python3
import logging

import numpy as np
import pandas as pd
import torch
from numpy import divide
from torch import sigmoid
from torch.utils.data import SequentialSampler

from dataset import split_train_test_set
from evaluation import multi_label_metrics


def predict(model, test_dataset, ids_to_labels):
    test = test_dataset
    test_dataloader = torch.utils.data.DataLoader(test, sampler=SequentialSampler(test_dataset), batch_size=4)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    pred_labels = [] # 'HP:000008'
    f_score_list = []
    p_list = []
    r_list = []
    logging.info("start predicting ...")

    with torch.no_grad():
        for input_ids, token_type_ids, attention_masks, b_labels in test_dataloader:
            mask = attention_masks.to(device)
            token_type_ids = token_type_ids.to(device)
            input_id = input_ids.squeeze(1).to(device)
            b_labels = b_labels.to(device)
            logits = model(input_ids=input_id, attention_mask=mask, token_type_ids=token_type_ids)

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            output, pred_y = predict_res_logits(logits, ids_to_labels, 0.35)
            pred_labels += output

            metrics = multi_label_metrics(logits, label_ids, threshold=0.35)
            f_score_list.append(metrics['f1'])
            p_list.append(metrics['precision'])
            r_list.append(metrics['recall'])
    print(f_score_list)
    print(p_list)
    print(r_list)
    # calculate avg performance
    avg_f = np.mean(f_score_list)
    avg_p = np.mean(p_list)
    avg_r = np.mean(r_list)
    print(avg_f)
    print(avg_p)
    print(avg_r)
    logging.info('avg f: %s' % str(avg_f))
    logging.info('avg p: %s' % str(avg_p))
    logging.info('avg r: %s' % str(avg_r))

    return pred_labels


def predict_res_logits(probs, ids_to_labels, threshold=0.35):
    if np.array(probs).max() < threshold:
        threshold = np.array(probs).mean()
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    pred_labels = []
    for _, row in enumerate(y_pred):
        labels = []
        for i in range(len(row)):
            if row[i] == 1:
                labels.append(ids_to_labels[i])
        pred_labels.append(labels)
    return pred_labels, y_pred


def predict_res_trainer(trainer, test_dataset, threshold, ids_to_labels):
    pred = trainer.predict(test_dataset=test_dataset).predictions
    probs = sigmoid(pred)
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    pred_labels = []
    for _, row in enumerate(y_pred):
        labels = []
        for i in range(len(row)):
            if row[i] == 1:
                labels.append(ids_to_labels[i])
        pred_labels.append(labels)
    return pred_labels


def save_result(save_path, pred_labels, df):
    # pred_labels = predict_res(trainer, test_dataset, 0.6, ids_to_labels)
    _, _, test_df = split_train_test_set(df)
    new_df = pd.DataFrame()
    for _, i in enumerate(test_df.indices):
        new_df = pd.concat([new_df, df.loc[i]], axis=1)
        # new_df = new_df.append(df.loc[i], ignore_index=True)
    # new_df['pred_labels'] = pred_labels
    test = pd.DataFrame(new_df.values.T, columns=new_df.index, index=new_df.columns)
    test['pred_labels'] = pred_labels
    test.to_csv(save_path, sep='\t', index=False)


def get_p_r_f_arrary(test_predict_label, test_true_label):
    test_predict_label = np.array(test_predict_label)
    test_true_label = np.array(test_true_label)
    num, cat = test_predict_label.shape
    acc_list = []
    prc_list = []
    rec_list = []
    f_score_list = []
    for i in range(num):
        label_pred_set = set()
        label_gold_set = set()

        for j in range(cat):
            if test_predict_label[i, j] == 1:
                label_pred_set.add(j)
            if test_true_label[i, j] == 1:
                label_gold_set.add(j)

        uni_set = label_gold_set.union(label_pred_set)
        intersec_set = label_gold_set.intersection(label_pred_set)

        tt = len(intersec_set)
        if len(label_pred_set) == 0:
            prc = 0
        else:
            prc = tt / len(label_pred_set)

        acc = tt / len(uni_set)

        rec = tt / len(label_gold_set)

        if prc == 0 and rec == 0:
            f_score = 0
        else:
            f_score = 2 * prc * rec / (prc + rec)

        acc_list.append(acc)
        prc_list.append(prc)
        rec_list.append(rec)
        f_score_list.append(f_score)

    mean_prc = np.mean(prc_list)
    mean_rec = np.mean(rec_list)
    f_score = divide(2 * mean_prc * mean_rec, (mean_prc + mean_rec))
    return mean_prc, mean_rec, f_score


if __name__ == '__main__':
    pass