#!/usr/bin/python3
from torch import nn
from transformers import BertModel


class BertClassifier(nn.Module):

    def __init__(self, class_num, dropout, hidden_size, model_name='bert-base-cased'):
        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, class_num)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits

