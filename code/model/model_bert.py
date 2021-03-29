import torch
import os
from transformers.models.bert.modeling_bert import BertModel, BertForMaskedLM, BertForSequenceClassification
from transformers.models.bert import BertTokenizer, BertConfig
from torch.nn import Module, Linear, Dropout, ReLU, Sequential, Parameter
from torch.nn.init import uniform_
from torch.nn.init import xavier_normal_, kaiming_normal_, normal_, constant_
from torch.nn.functional import log_softmax, relu, softmax
from torch.nn import CrossEntropyLoss, MSELoss, AdaptiveAvgPool1d, AdaptiveAvgPool2d, LSTM, Embedding
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import RobertaForSequenceClassification, BertPreTrainedModel
from torch import nn
import torch.nn.functional as F
from nezha import modeling_nezha as nezha
from transformers.modeling_outputs import MaskedLMOutput


class MyBertForSequenceClassification10(nezha.BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(MyBertForSequenceClassification10, self).__init__(config)
        self.num_labels = num_labels
        self.bert = nezha.BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                     output_all_encoded_layers=False)
        task_output = self.dropout(pooled_output)
        logits = self.classifier(task_output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return MaskedLMOutput(loss=loss, logits=logits)
        else:
            return MaskedLMOutput(loss=None, logits=logits)

