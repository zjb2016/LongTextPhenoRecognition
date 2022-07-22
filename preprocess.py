#!/usr/bin/python3
import logging
import re

import seaborn as sns
import numpy as np
import pandas as pd
import os

from matplotlib import pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# from transformers import BertTokenizer
from transformers import BertTokenizer


def load_data(path):
    df = pd.read_csv(path, sep='\t', encoding='utf-8')
    unique_label_len, labels_to_ids, ids_to_labels = build_label_dict(df)
    df.loc[:, 'label_code'] = df.apply(encode_label, args=(unique_label_len, labels_to_ids), axis=1)
    # process labels
    labels = np.array(df['label_code'].values.tolist())
    return df, labels, labels_to_ids, ids_to_labels



def build_label_dict(df):
    # Split labels based on whitespace and turn them into a list
    labels = []
    for i in df['labels'].values.tolist():
        if pd.notna(i):
            labels.extend([m.strip() for m in re.split(";|,", i)])

    # Check how many labels are there in the dataset
    unique_labels = set(labels)

    # Map each label into its id representation and vice versa
    labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
    ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}

    logging.info('Class num: %d ' % len(unique_labels))
    logging.info('Class: %s' % ids_to_labels)
    return len(unique_labels), labels_to_ids, ids_to_labels


def generate_label_list(labels, unique_label_len, labels_to_ids):
    label_list = np.zeros(unique_label_len)
    if isinstance(labels, str):
        # print(labels)
        arr = [m.strip() for m in re.split(";|,", labels)]
        for label in arr:
            label_list[labels_to_ids[label]] = 1
    return label_list


def encode_label(df, unique_label_len, labels_to_ids):
    return generate_label_list(df['labels'], unique_label_len, labels_to_ids)


def str2arr(str):
    list1 = str[1:-1].split('.')
    list2 = []
    for j in list1:
        if len(j) > 0:
            list2.append(float(j))
    arr1 = np.array(list2)
    return arr1


def text_length_check(sentences, tokenizer):
    avg_len = 0
    all = 0
    max_len = 0
    cnt = 0
    over_index = []
    # For every sentence...
    for sent in sentences:
        # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
        input_ids = tokenizer.encode(sent, add_special_tokens=True)
        all+=len(input_ids)
        if len(input_ids) > 512:
            # print(cnt)
            over_index.append(cnt)
            cnt = cnt + 1
        # Update the maximum sentence length.
        max_len = max(max_len, len(input_ids))
        avg_len = all/len(sentences)

    print('Max sentence length: ' + str(max_len))
    print('avg sentence length: ' + str(avg_len))
    print('Over length count: ' + str(cnt))
    logging.info('Max sentence length: ' + str(max_len))
    logging.info('Over length count: ' + str(cnt))


def calculate_weight_for_imbalanced_data(labels, ids_to_labels):
    x = np.sum(labels, axis=0).astype('int')
    # draw distribution
    data = pd.DataFrame()
    data['Count'] = list(x)
    data['HPO_ID'] = ids_to_labels.values()
    plt.figure(figsize=(35, 20))
    plt.title('Layer-1 HPO Nodes distribution on GSC+', fontsize=32)

    b=sns.barplot(data=data, x='HPO_ID', y='Count',palette=sns.set_context("paper", rc={"font.size": 48, "axes.titlesize": 48, "axes.labelsize": 48}))
    b.set_ylabel('Count', size=32)
    b.set_xlabel('HPO_ID', size=32)
    plt.savefig('data_dis.jpg')
    plt.show()

    # calculate weights
    class_weights = []
    class_sum = np.sum(x)
    for label_cnt in x:
        weight = 1-label_cnt/class_sum
        class_weights.append(weight)
    return class_weights


if __name__ == '__main__':
    df, labels, labels_to_ids, ids_to_labels = load_data('data/gsc.csv')
    calculate_weight_for_imbalanced_data(labels, ids_to_labels)
    tokenizer = BertTokenizer.from_pretrained("D://summerTerm//pretrain_models//bluebert_pubmed_uncased", lowercase=True)
    text_length_check(df['text'],tokenizer)
    print(df.sample(5))
