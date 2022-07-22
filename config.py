#!/usr/bin/python3
import logging
import sys

MAX_LEN = 512
UNIQUE_LABEL_LEN = 217
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
EPOCHS = 1
MODEL_PATH = "model.bin"
MODEL_SAVE_PATH = 'output/'
LR = 5e-5
EPS = 1e-8
DROPOUT = 0.1
HIDDEN_SIZE = 768
IS_LOWER_CASE = False
PATIENCE = 4
LOAD_SAVED_MODEL = False

# MODEL_NAME = 'bert-base-cased'
# MODEL_NAME = 'dmis-lab/biobert-base-cased-v1.1'
# MODEL_NAME = 'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12'
MODEL_NAME = "D://summerTerm//pretrain_models//bluebert_pubmed_uncased"
# TRAINING_FILE = 'D://summerTerm//blue_benchmark_data//data//hoc//train.tsv'
TRAINING_FILE = "data/gsc.csv"


def setup_log(file):
    logging.basicConfig(filename=file, level=logging.INFO, filemode='w')
    logging.raiseExceptions = False  # cancel error output stdout
    logging.FileHandler(file, encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')

    logging.info('******* Program start ********\n')
    logging.info('Dataset: %s' % TRAINING_FILE)
    logging.info('Pretrain Model: %s' % MODEL_NAME)


def print_model_params(model):
    logging.info('\n ====== The BERT model structure ====== \n')
    logging.info(model)

    # Get all the model's parameters as a list of tuples.
    params = list(model.named_parameters())
    logging.info('The BERT model has {:} different named parameters.\n'.format(len(params)))
    logging.info('==== Embedding Layer ====\n')

    for p in params[0:5]:
        logging.info("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    logging.info('\n==== First Transformer ====\n')

    for p in params[5:21]:
        logging.info("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    logging.info('\n==== Output Layer ====\n')

    for p in params[-4:]:
        logging.info("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
