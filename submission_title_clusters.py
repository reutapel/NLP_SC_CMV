import pandas as pd
import os
import joblib
import time
from doc2vec import Doc2Vec
import torch
from pytorch_transformers import *
import datetime
from tqdm import *
from submissions_clusters import *
# PyTorch-Transformers has a unified API
# for 6 transformer architectures and 27 pretrained weights.
# #          Model          | Tokenizer          | Pretrained weights shortcut
# MODELS = [(BertModel,       BertTokenizer,      'bert-base-uncased'),
#           (OpenAIGPTModel,  OpenAIGPTTokenizer, 'openai-gpt'),
#           (GPT2Model,       GPT2Tokenizer,      'gpt2'),
#           (TransfoXLModel,  TransfoXLTokenizer, 'transfo-xl-wt103'),
#           (XLNetModel,      XLNetTokenizer,     'xlnet-base-cased'),
#           (XLMModel,        XLMTokenizer,       'xlm-mlm-enfr-1024')]

    # # Each architecture is provided with several class for fine-tuning on down-stream tasks, e.g.
    # BERT_MODEL_CLASSES = [BertModel, BertForPreTraining, BertForMaskedLM, BertForNextSentencePrediction,
    #                   BertForSequenceClassification, BertForMultipleChoice, BertForTokenClassification,
    #                   BertForQuestionAnswering]
    #
    # # All the classes for an architecture can be initiated from pretrained weights for this architecture
    # # Note that additional weights added for fine-tuning are only initialized
    # # and need to be trained on the down-stream task
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # for model_class in BERT_MODEL_CLASSES:
    #     # Load pretrained model/tokenizer
    #     model = BertModel.from_pretrained('bert-base-uncased')
    #
    # # Models can return full list of hidden-states & attentions weights at each layer
    # model = BertModel.from_pretrained('bert-base-uncased',
    #                                     output_hidden_states=True,
    #                                     output_attentions=True)
    # input_ids = torch.tensor([tokenizer.encode("Let's see all hidden-states and attentions on this text")])
    # all_hidden_states, all_attentions = model(input_ids)[-2:]
    #
    # # Models are compatible with Torchscript
    # model = BertModel.from_pretrained('bert-base-uncased', torchscript=True)
    # traced_model = torch.jit.trace(model, (input_ids,))
    #
    # # Simple serialization for models and tokenizers
    # model.save_pretrained('./directory/to/save/')  # save
    # model = BertModel.from_pretrained('./directory/to/save/')  # re-load
    # tokenizer.save_pretrained('./directory/to/save/')  # save
    # tokenizer = BertTokenizer.from_pretrained('./directory/to/save/')  # re-load
    # return


class SubmissionsTitleClusters:
    """ creates embedded version of submission titles and clusters them """

    def __init__(self, data, data_directory, embedding_size, take_bert_pooler=True):

        self.data = data
        self.data_directory = data_directory

        # embedding class variables
        self.embedding_size = embedding_size
        self.doc2vec_model = None
        self.doc2vec_fitted_model_file_path = None
        self.bert_encoded_submissions_title_df = pd.DataFrame(columns=['embedded_tokens', 'pooler'])
        self.take_bert_pooler = take_bert_pooler

    def bert_encoding(self, is_save=True):
        """
        encode all submission titles using BERT model encoder
        :return: fills class variable df bert_encoded_submissions_title_df
        """
        # Load pretrained model/tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')

        # Encode text
        if self.take_bert_pooler:
            print('starting BERT encoding', datetime.datetime.now())
            for index, row in tqdm(self.data.iteritems(), total=self.data.shape[0]):
                # print(len(row.split(" ")), ' number of words')
                # print(len(set(row.split(" "))), ' number of DISTINCT words')
                tokenized = tokenizer.tokenize(row)
                # print(len(tokenized), ' number of BERT tokens')

                # add dummy token for pooler - sum of sentence
                tokenized.insert(0, 'CLS')
                input_ids = tokenizer.convert_tokens_to_ids(tokenized)
                # print(len(input_ids), ' number of BERT ids')
                # print('first BERT id for CLS dummy token is: ', input_ids[0])
                # input_ids = torch.tensor([tokenizer.encode(row)])
                with torch.no_grad():
                    pooler = model(torch.tensor([input_ids]))[0][0][0]
                    # print(pooler)
                    self.bert_encoded_submissions_title_df.loc[index, 'pooler'] = pooler
            print('finished BERT encoding', datetime.datetime.now())
            if is_save:
                print('saving BERT embedded df')
                joblib.dump(self.bert_encoded_submissions_title_df, 'self.bert_encoded_submissions_title_df.pickle')
        return

    def doc2vec_embedding(self, min_count=2, epochs=200):
        """
        embed using doc2vec
        :return:
        """
        # fit embedding by doc2vec
        print('{}: starting fitting doc2vec model'.format(time.asctime(time.localtime(time.time()))))
        self.doc2vec_model = Doc2Vec(fname='', linux=False, use_file=False, data=self.data,
                                     vector_size=self.embedding_size, min_count=min_count, epochs=epochs)
        print('{}: finish fitting doc2vec model'.format(time.asctime(time.localtime(time.time()))))

        # save trained model
        self.doc2vec_fitted_model_file_path = os.path.join(self.data_directory, 'doc2vec_submission_titles.pkl')
        print('{}: starting saving doc2vec model'.format(time.asctime(time.localtime(time.time()))))
        joblib.dump(self.doc2vec_model, self.doc2vec_fitted_model_file_path)
        print('{}: finishing saving doc2vec model'.format(time.asctime(time.localtime(time.time()))))

        return

    def describe_data(self):

        print('{}: analyzing data '.format(time.asctime(time.localtime(time.time()))))
        data_len = self.data.apply(lambda x: len(x.split(' ')))
        print('describing length of titles statistics: ')
        print(data_len.describe())

        data_splitted = self.data.apply(lambda x: x.split(' '))
        corpus = list(set([a for b in data_splitted.tolist() for a in b]))
        print('unique words count is: ', len(corpus))


def main():

    # define paths
    base_directory = os.path.abspath(os.curdir)
    data_directory = os.path.join(base_directory, 'features_to_use')

    # load data
    data_file_path = os.path.join(data_directory, 'all_train_data.csv')
    all_train_data = pd.read_csv(data_file_path)
    print("raw data shape: ", all_train_data.shape)

    # take only submission titles
    all_train_data_submission_title = all_train_data['submission_title'].copy()
    all_train_data_submission_title_unique = all_train_data_submission_title.unique()
    print("unique data shape: ", all_train_data_submission_title_unique.shape)
    del all_train_data

    # remove CMV prefix from titles that have it
    all_train_data_submission_title_unique = \
        pd.Series(all_train_data_submission_title_unique).apply(lambda x: x[5:] if x.startswith('CMV:') else x)

    # create class obj
    sub_title_cluster_obj = SubmissionsTitleClusters(data=all_train_data_submission_title_unique, embedding_size=300,
                                                     data_directory=data_directory, take_bert_pooler=True)
    sub_title_cluster_obj.describe_data()

    # encode
    # sub_title_cluster_obj.doc2vec_embedding(min_count=2, epochs=1000)
    sub_title_cluster_obj.bert_encoding(is_save=True)

    # # evaluate embedding
    # # define number of doc to test doc2vec embedding
    # num_of_docs_to_test = 5
    # sub_title_cluster_obj.doc2vec_model.evaluate_self_sim(num_of_docs_to_test)

    # cluster embedded data
    submissions_clusters_obj = SubmissionsClusters(sub_title_cluster_obj.bert_encoded_submissions_title_df.loc[:,
                                                   'pooler'])

    submission_title_bert_embedded_x_tsne = submissions_clusters_obj.tsne_cluster()
    joblib.dump(submission_title_bert_embedded_x_tsne, 'submission_title_bert_embedded_x_tsne.pickle')
    submission_title_bert_embedded_y_gmm = submissions_clusters_obj.gmm_cluster()
    joblib.dump(submission_title_bert_embedded_y_gmm, 'submission_title_bert_embedded_y_gmm.pickle')


if __name__ == '__main__':
    main()

