import torch
from pytorch_transformers import *
import pandas as pd
import datetime
import numpy as np

# # for 6 transformer architectures and 27 pretrained weights.
# #          Model          | Tokenizer          | Pretrained weights shortcut
# MODELS = [(BertModel,       BertTokenizer,      'bert-base-uncased'),
#           (OpenAIGPTModel,  OpenAIGPTTokenizer, 'openai-gpt'),
#           (GPT2Model,       GPT2Tokenizer,      'gpt2'),
#           (TransfoXLModel,  TransfoXLTokenizer, 'transfo-xl-wt103'),
#           (XLNetModel,      XLNetTokenizer,     'xlnet-base-cased'),
#           (XLMModel,        XLMTokenizer,       'xlm-mlm-enfr-1024')]
#
#     # Each architecture is provided with several class for fine-tuning on down-stream tasks, e.g.
#     BERT_MODEL_CLASSES = [BertModel, BertForPreTraining, BertForMaskedLM, BertForNextSentencePrediction,
#                       BertForSequenceClassification, BertForMultipleChoice, BertForTokenClassification,
#                       BertForQuestionAnswering]


class BertTransformer:
    """ creates embedded version of submission titles and clusters them """

    def __init__(self, pretrained_weights_shortcut='bert-base-uncased', take_bert_pooler=True):

        self.Pretrained_weights_shortcut = pretrained_weights_shortcut
        self.take_bert_pooler = take_bert_pooler
        self.poolers_df = None

        # TODO: change to be able to choose which model / tokenizer for downstream tasks
        # Load pretrained model/tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.Pretrained_weights_shortcut)
        self.model = BertModel.from_pretrained(self.Pretrained_weights_shortcut)

    def bert_text_encoding(self, text, take_bert_pooler=True):
        """
        embedd text with BERT pre trained model and tokenizer that is defined in init
        :param text: text to embedd
        :param take_bert_pooler: take pooler - sum of sentence or rest of embedded tensor for sentence tokens
        :return: requested embedded part
        """
        # Encode text
        print('starting BERT encoding', datetime.datetime.now())
        tokenized = self.tokenizer.tokenize(text)
        tokenized.insert(0, 'CLS')
        input_ids = self.tokenizer.convert_tokens_to_ids(tokenized)
        with torch.no_grad():
            if take_bert_pooler:
                pooler = self.model(torch.tensor([input_ids]))[0][0][0]
                print('finished BERT encoding', datetime.datetime.now())
                return pooler
            else:
                # TODO: validate tokens [1:]
                tokens_embedding = self.model(torch.tensor([input_ids]))[0][0][1:]
                print('finished BERT encoding', datetime.datetime.now())
                return tokens_embedding

    def average_vectors(self, tensors_list):
        """ assumes tensors are in the same dimensions"""
        return np.mean(tensors_list, axis=0)

    def split_text_by_max_size(self, text, max_size):
        """
        splits text to parts built from complete sentences , wheres part is maximum max_size number of words
        :param text:
        :param max_size:
        :return: list of parts of text
        """
        # list of text sub strings of maximum length of max_size
        text_parts_list = list()
        # assuming sentences in text are divided by '.'
        sentence_list = text.split('.')
        # current part of text to concatenate sentences
        current_part = str()
        # length of current part
        current_part_len = 0
        while sentence_list:
            # take next sentence
            current_sentence = sentence_list.pop(0)
            # check if empty string
            if len(current_sentence) == 0:
                continue
            # length of current sentence
            current_tokens_count = len(current_sentence.split(' '))
            if current_part_len + current_tokens_count <= max_size:
                current_part = current_part + current_sentence
                current_part_len += current_tokens_count
            else:
                text_parts_list.append(current_part)
                current_part = str()
                current_part_len = 0
        # in case entire text never crossed max_size or last part didn't..
        if current_part in text_parts_list:
            return text_parts_list
        else:
            text_parts_list.append(current_part)
            return text_parts_list

    def get_text_average_pooler(self, text, max_size, take_bert_pooler=True):
        """
        this method gets a text, splits it to allowed size, encode, gets the poolers and average them
        :param text:
        :param max_size:
        :param take_bert_pooler:
        :return: average pooler
        """
        poolers_list = list()
        # get splited text
        text_parts_list = self.split_text_by_max_size(text, max_size)
        # encode each text part
        for text_part in text_parts_list:
            text_part_pooler = self.bert_text_encoding(text_part, take_bert_pooler)
            poolers_list.append(pd.Series(text_part_pooler))
            # poolers_df = pd.DataFrame(poolers_list)
        # average all poolers of text
        average_pooler = self.average_vectors(poolers_list)
        return average_pooler





