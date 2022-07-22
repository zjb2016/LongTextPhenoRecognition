#!/usr/bin/python3
from __future__ import print_function

import torch
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup

import config
from dataset import generate_dataset, split_train_test_set, create_dataloader
from evaluation import summary_train, draw_train_stats
from model import BertClassifier
from preprocess import load_data, text_length_check, calculate_weight_for_imbalanced_data
from test import predict, save_result
from train import train_model

epochs = config.EPOCHS
eps = config.EPS
lr = config.LR
model_name = config.MODEL_NAME
dropout = config.DROPOUT
hidden_size = config.HIDDEN_SIZE
is_lower_case = config.IS_LOWER_CASE


config.setup_log('logger.log')

df, labels, labels_to_ids, ids_to_labels = load_data(config.TRAINING_FILE)
print('Loading %d data item' % len(df))

tokenizer = BertTokenizer.from_pretrained(model_name, lowercase=is_lower_case)
print('Loading bert tokenizer, lower case is ', str(is_lower_case))

sentences = df['text']
dataset = generate_dataset(sentences, tokenizer, labels)
text_length_check(sentences,tokenizer)
train_dataset, val_dataset, test_dataset = split_train_test_set(dataset)

i = 0
print("\n ***  encode examples  ***")
print("sentence: %s" % " ".join(sentences[i]))
print("tokens: %s" % " ".join(tokenizer.convert_ids_to_tokens(dataset[i][0])))
print("input_ids: %s" % dataset[i][0])
print("token_type_ids: %s" % dataset[i][1])
print("attention_mask: %s" % dataset[i][2])
print("label: %s " % labels[i])

print('split the train, validation and test set:')
print(len(train_dataset), len(val_dataset), len(test_dataset))

print('create dataloader...')
train_dataloader, validation_dataloader = create_dataloader(train_dataset, val_dataset)

print('load the model...')
class_num = len(labels_to_ids)
model = BertClassifier(class_num, dropout, hidden_size, model_name=model_name)
if config.LOAD_SAVED_MODEL:
    model.load_state_dict(torch.load("output/best_network.pth", map_location=torch.device('cpu')))
config.print_model_params(model)


# define the optimizer
optimizer = AdamW(model.parameters(),
                  lr=lr,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps=eps  # args.adam_epsilon  - default is 1e-8.
                  )
# Total number of training steps is [number of batches] x [number of epochs].
total_steps = len(train_dataloader) * epochs
print('Training steps:', total_steps)

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,  # Default value in run_glue.py
                                            num_training_steps=total_steps)

print('start training...')
weights = calculate_weight_for_imbalanced_data(labels, ids_to_labels)
training_stats = train_model(model, train_dataloader, validation_dataloader, scheduler, optimizer, epochs, weights)

stats = summary_train(training_stats)
draw_train_stats(stats, epochs)

print('start predicting...')
pred_labels  = predict(model, test_dataset, ids_to_labels)
save_result('predict.csv', pred_labels, df)
