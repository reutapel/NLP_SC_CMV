import pandas as pd
from time import time
import time
import math
from datetime import datetime
import logging
import pytz
import copy
import os
import nltk as nk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim.sklearn_api import ldamodel
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from doc2vec import Doc2Vec
import joblib
import sys
import ray
import re
# from gensim import corpora
# from nltk.stem import PorterStemmer
# from nltk.tokenize import sent_tokenize, word_tokenize


base_directory = os.getenv('PWD')
data_directory = os.path.join(base_directory, 'data')
trained_models_directory = os.path.join(base_directory, 'trained_models')
save_data_directory = os.path.join(data_directory, 'filter_submissions')
train_test_data_directory = os.path.join(data_directory, 'filter_submissions')
features_directory = os.path.join(base_directory, 'features_to_use')
log_directory = os.path.join(base_directory, 'logs')


# for topic modeling clean text
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
# for sentiment analysis
sid = SentimentIntensityAnalyzer()

max_branch_length_dict = {
    'train': 108,
    'testi': 102,
    'valid': 106,
}


def save_as_pickled_object(obj, file_path):
    """
    This is a defensive way to write pickle.write, allowing for very large files on all platforms
    """
    max_bytes = 2**31 - 1
    bytes_out = pickle.dumps(obj)
    n_bytes = sys.getsizeof(bytes_out)
    with open(file_path, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])

    return


class CreateFeatures:
    """
    This class will build the features for each unit: comment and its submission
    """
    def __init__(self, number_of_topics, max_branch_length=None):
        """
        Load data and pre process
        :param int number_of_topics: number_of_topics for topic model feature
        :param int max_branch_length: if there is a specific max_branch_length
        """

        self.is_train = None
        self.data_file_name = None
        self.number_of_topics = number_of_topics

        # Load branches data
        branch_columns = ['branch_id', 'num_delta', 'submission_id', 'branch_length', 'delta_index_in_branch',
                          'num_comments_after_delta']
        self.branch_numbers_df = pd.read_csv(os.path.join(save_data_directory,
                                                          'comments_label_branch_info_after_remove.csv'),
                                             usecols=branch_columns)
        self.branch_numbers_df = self.branch_numbers_df.drop_duplicates(subset='branch_id')

        # self.branch_numbers_df.columns = ['branch_id', 'branch_key', 'num_delta', 'submission_id',
        #                                   'num_comments_removed', 'pct_comments_removed', 'branch_length',
        #                                   'delta_index_in_branch', 'num_comments_after_delta']
        self.all_branches = copy.deepcopy(self.branch_numbers_df)

        # Create submissions data:
        self.all_submissions_total = pd.read_csv(os.path.join(save_data_directory,
                                                              'all_submissions_final_after_remove.csv'))
        self.all_submissions = None
        self.all_data_set_submissions = None

        # Load all relevant data

        self.data_columns = ['comment_author', 'comment_body', 'comment_created_utc', 'comment_id', 'comment_is_root',
                             'comment_is_submitter', 'delta', 'parent_id', 'submission_id', 'branch_id',
                             'comment_real_depth', 'branch_length', 'submission_created_utc', 'submission_body',
                             'submission_author', 'submission_title']

        self.data = None
        self.all_data = None

        # define branch_comments_raw_text_df with number of columns as the max_branch_length
        self.branch_comments_embedded_text_df = None

        # Create comments_features
        self.comment_features_columns = ['comment_real_depth', 'comment_len', 'time_between_sub_com',
                                         'percent_adj', 'time_between_comment_first_comment', 'submission_num_comments',
                                         'time_ratio_first_comment', 'nltk_com_sen_pos', 'nltk_com_sen_neg',
                                         'nltk_com_sen_neutral', 'nltk_sim_sen', 'is_quote', 'number_of_branches',
                                         'num_mentioned_subreddit', 'num_mentioned_url']
        self.comment_features_columns_len = len(self.comment_features_columns) + number_of_topics
        self.branch_comments_features_df = None

        # Create comments_features
        self.comments_user_features_columns = ['commenter_number_submission',
                                               'number_of_comments_in_submission_by_comment_user',
                                               'commenter_seniority_days', 'is_first_comment_in_tree',
                                               'submmiter_commenter_tfidf_cos_sim', 'respond_to_comment_user_all_ratio',
                                               'respond_to_comment_user_responses_ratio', 'commenter_number_comment',
                                               'number_of_respond_by_submitter_to_commenter']
        self.branch_comments_user_profiles_df = None

        # Create submission features
        self.submission_features_columns = ['submission_len', 'title_len', 'time_until_first_comment',
                                            'nltk_sub_sen_pos', 'nltk_sub_sen_neg', 'nltk_sub_sen_neutral',
                                            'nltk_title_sen_pos', 'nltk_title_sen_neg', 'nltk_title_sen_neutral',
                                            'comments_in_submission']

        # Create submission user features
        self.submitter_features_columns = ['submitter_number_submission', 'submitter_number_comment',
                                           'submitter_seniority_days', 'number_of_comments_in_tree_from_submitter',
                                           'comments_total_ratio']

        # Create branch_submission_dict
        self.branch_submission_dict_features = ['branch_length', 'num_deltas_in_submission_before_branch',
                                                'num_brother_branches_before_branch', 'mean_brothers_length']

        self.max_branch_length = max_branch_length
        # define class
        self.branch_ids = None
        self.branches_lengths_list = None
        self.num_branches = None
        self.submission_no_dup = None
        self.vocab_df = None
        self.tfidf_vec_fitted = None
        self.train_dictionary = None
        self.train_data_term_matrix = None
        self.lda_model = None
        self.doc2vec_model = None
        self.doc2vec_vector_size = 200
        self.submission_comments_dict = None
        self.branch_comments_dict = None
        self.submission_data_dict = dict()
        self.branch_deltas_data_dict = dict()
        self.branch_submission_dict = dict()

        return

    def create_data(self, data_file_name, is_train, load_data=True, data=None, data_dir=train_test_data_directory,
                    trained_models_dir=trained_models_directory, all_data=None):
        """
        This function create the data based on the data_file_name
        :param str data_file_name: the data to create (train, test, val)
        :param bool is_train: if this is the train data
        :param data: the data itself, if already load it
        :param data_dir: if we load from directory which is not the main one
        :param bool load_data: if we want to load the data or no need to
        :param trained_models_dir: the path of the trained models directory
        :param all_data: all the data of the data set (train, test, val)- no split data
        :return:
        """
        self.data_file_name = data_file_name
        self.is_train = is_train

        if load_data:
            # all the data, with label
            if data is None:
                print(f'{time.asctime(time.localtime(time.time()))}: Loading data {data_file_name} from {data_dir}')
                logging.info('Loading data {} from {}'.format(data_file_name, data_dir))

                self.data = pd.read_csv(os.path.join(data_dir, data_file_name + '_data.csv'), skipinitialspace=True,
                                        usecols=self.data_columns)
                self.data['comment_created_utc'] = self.data['comment_created_utc'].astype(int)
                self.data['time_between'] = self.data['comment_created_utc'] - self.data['submission_created_utc']

                # Features calculated for all the data frame:
                self.data['comment_len'] = self.data['comment_body'].str.len()

                # get number of branches for each comment
                comment_branch = self.data[['comment_id', 'branch_id']]
                comment_branch_groupby = comment_branch.groupby(by='comment_id').count()
                comment_branch_groupby['comment_id'] = comment_branch_groupby.index
                comment_branch_groupby.columns = ['number_of_branches', 'comment_id']

                self.data = self.data.merge(comment_branch_groupby, on='comment_id')
                # 
                # load and do the pre process on all_data
                print(f'{time.asctime(time.localtime(time.time()))}: Loading all data {data_file_name} from {data_dir}')
                logging.info('Loading all data {} from {}'.format(data_file_name, data_dir))
                self.all_data = pd.read_csv(os.path.join(data_dir, 'all_' + data_file_name + '_data.csv'),
                                            skipinitialspace=True, usecols=self.data_columns)
                self.all_data['comment_created_utc'] = self.all_data['comment_created_utc'].astype(int)
                self.all_data['time_between'] = self.all_data['comment_created_utc'] -\
                                                self.all_data['submission_created_utc']

                # Features calculated for all the data frame:
                self.all_data['comment_len'] = self.all_data['comment_body'].str.len()

                # get number of branches for each comment
                comment_branch = self.all_data[['comment_id', 'branch_id']]
                comment_branch_groupby = comment_branch.groupby(by='comment_id').count()
                comment_branch_groupby['comment_id'] = comment_branch_groupby.index
                comment_branch_groupby.columns = ['number_of_branches', 'comment_id']

                self.all_data = self.all_data.merge(comment_branch_groupby, on='comment_id')

            else:
                self.data = data
                self.all_data = all_data

            self.data_pre_process()

            # self.data['submission_created_utc'] = self.data['submission_created_utc'].astype(int)
            self.all_submissions['submission_created_utc'] = self.all_submissions['submission_created_utc'].astype(int)
            self.all_data_set_submissions['submission_created_utc'] =\
                self.all_data_set_submissions['submission_created_utc'].astype(int)

            # get the max number of comments in a branch
            if self.max_branch_length is None:  # if we didn't define this length before
                self.max_branch_length = self.all_branches['branch_length'].max()

            # get the branch ids sort by their length from the longest to the shortest
            self.branch_ids = self.all_branches[['branch_id', 'branch_length']].drop_duplicates()
            self.branch_ids = self.branch_ids.sort_values(by='branch_length', ascending=False)
            self.branches_lengths_list = list(self.branch_ids['branch_length'])
            self.num_branches = len(self.branches_lengths_list)
            self.branch_ids = self.branch_ids.reset_index()
            self.branch_ids = self.branch_ids['branch_id']

            # fill features df with list of 0 according to the max branch length
            self.branch_comments_features_df = dict()
            self.branch_comments_user_profiles_df = dict()

            # define branch_comments_raw_text_df with number of columns as the max_branch_length
            # self.branch_comments_embedded_text_df = pd.DataFrame(columns=np.arange(self.max_branch_length))
            self.branch_comments_embedded_text_df = dict()

            # Create vocabulary of the text of the data
            # data after drop duplications:
            self.submission_no_dup = self.all_submissions[['submission_author', 'submission_id',
                                                           'submission_created_utc']]

            print("{}: Start create vocab".format(time.asctime(time.localtime(time.time()))))
            logging.info('Start create vocab')
            self.vocab_df = self.create_vocab()
            print("{}: Finish create vocab".format(time.asctime(time.localtime(time.time()))))
            logging.info('Finish create vocab')

            # create dict with the data for each submission:
            submission_comments_dict_file_path =\
                os.path.join(data_dir, 'trained_models', 'submission_comments_dict_' + self.data_file_name + '.pkl')
            if not os.path.isfile(submission_comments_dict_file_path):
                print('{}: Start create submission dict for {}'.format((time.asctime(time.localtime(time.time()))),
                                                                       self.data_file_name))
                logging.info('Start create submission dict for {}'.format(self.data_file_name))
                start_time = time.time()
                self.submission_comments_dict = dict()
                submission_ids = self.submission_no_dup['submission_id']
                for index, submission_id in submission_ids.iteritems():
                    self.submission_comments_dict[submission_id] =\
                        self.all_data.loc[self.all_data['submission_id'] == submission_id].\
                            drop_duplicates(subset='comment_id')

                joblib.dump(self.submission_comments_dict, submission_comments_dict_file_path)
                print('time to create submission dict: ', time.time() - start_time)
                logging.info('time to create submission dict: {}'.format(time.time() - start_time))
            else:
                self.submission_comments_dict = joblib.load(submission_comments_dict_file_path)

            # create dict with the data for each branch:
            print('{}: Start create branch dict for {}'.format((time.asctime(time.localtime(time.time()))),
                                                               self.data_file_name))
            logging.info('Start create branch dict for {}'.format(self.data_file_name))
            self.all_branches = self.all_branches.assign(branch_first_comment=0)
            self.all_branches = self.all_branches.assign(branch_last_comment=0)
            start_time = time.time()
            self.branch_comments_dict = dict()
            for index, branch_id in self.branch_ids.iteritems():
                branch_data = self.data.loc[self.data['branch_id'] == branch_id].drop_duplicates(subset='comment_id')
                branch_data = branch_data.sort_values(by='comment_real_depth', ascending=True).reset_index()
                self.branch_comments_dict[branch_id] = branch_data
                # add first and last comment in branch
                self.all_branches.loc[self.all_branches.branch_id == branch_id, 'branch_first_comment'] =\
                    branch_data['comment_created_utc'].min()
                self.all_branches.loc[self.all_branches.branch_id == branch_id, 'branch_last_comment'] =\
                    branch_data['comment_created_utc'].max()
            print('time to create branch dict: ', time.time() - start_time)
            logging.info('time to create branch dict: {}'.format(time.time() - start_time))

        # create vocabulary for idf calq only if train data, else- use what we trained with the train data
        if self.is_train:
            tfidf_vec_fitted_file_path = os.path.join(trained_models_dir, 'tfidf_vec_fitted.pkl')
            if not os.path.isfile(tfidf_vec_fitted_file_path):
                print('{}: begin fitting tfidf'.format(time.asctime(time.localtime(time.time()))))
                logging.info('begin fitting tfidf')
                self.tfidf_vec_fitted = TfidfVectorizer(stop_words='english', lowercase=True, analyzer='word',
                                                        norm='l2', smooth_idf=True, sublinear_tf=False, use_idf=True)
                self.tfidf_vec_fitted.fit(self.vocab_df)
                joblib.dump(self.tfidf_vec_fitted, tfidf_vec_fitted_file_path)
                print('{}: finish fitting tfidf'.format(time.asctime(time.localtime(time.time()))))
                logging.info('finish fitting tfidf')

            else:  # load the trained model
                print('{}: Loading fitted tfidf'.format(time.asctime(time.localtime(time.time()))))
                logging.info('Loading fitted tfidf')
                self.tfidf_vec_fitted = joblib.load(tfidf_vec_fitted_file_path)

            # create LDA topic model
            train_dictionary_file_path = os.path.join(trained_models_dir, 'train_dictionary.pkl')
            train_data_term_matrix_file_path = os.path.join(trained_models_dir, 'train_data_term_matrix.pkl')

            if not os.path.isfile(train_dictionary_file_path):
                self.train_dictionary, self.train_data_term_matrix = self.create_data_dictionary()
                joblib.dump(self.train_dictionary, train_dictionary_file_path)
                joblib.dump(self.train_data_term_matrix, train_data_term_matrix_file_path)
            else:
                print('{}: Loading train_dictionary and train_data_term_matrix'.
                      format(time.asctime(time.localtime(time.time()))))
                logging.info('Loading train_dictionary and train_data_term_matrix')
                self.train_dictionary = joblib.load(train_dictionary_file_path)
                self.train_data_term_matrix = joblib.load(train_data_term_matrix_file_path)

            # Create LDA model
            lda_fitted_model_file_path = os.path.join(trained_models_dir, 'lda_model.pkl')
            if not os.path.isfile(lda_fitted_model_file_path):
                print('{}: Create LDA model'.format((time.asctime(time.localtime(time.time())))))
                logging.info('Create LDA model')

                self.lda_model = ldamodel.LdaTransformer(num_topics=self.number_of_topics,
                                                         id2word=self.train_dictionary,
                                                         passes=50, minimum_probability=0)
                # Train LDA model on the comments term matrix
                print(
                    '{}: Fit LDA model on {}'.format((time.asctime(time.localtime(time.time()))), self.data_file_name))
                logging.info('Fit LDA model on {}'.format(self.data_file_name))

                self.lda_model = self.lda_model.fit(list(self.train_data_term_matrix.values()))
                joblib.dump(self.lda_model, lda_fitted_model_file_path)
                print('{}: finish fitting LDA model'.format(time.asctime(time.localtime(time.time()))))
                logging.info('finish fitting LDA model')

            else:
                print('{}: Loading fitted LDA model'.format(time.asctime(time.localtime(time.time()))))
                logging.info('Loading fitted LDA model')
                self.lda_model = joblib.load(lda_fitted_model_file_path)

            # create and train doc2vec model
            doc2vec_fitted_model_file_path = os.path.join(trained_models_dir, 'doc2vec_model.pkl')
            if not os.path.isfile(doc2vec_fitted_model_file_path):
                print('{}: Create and train Doc2Vec model on {}'.format((time.asctime(time.localtime(time.time()))),
                                                                        self.data_file_name))
                logging.info('Create and train Doc2Vec model on {}'.format(self.data_file_name))
                submission_body = self.all_data_set_submissions['submission_body']
                comments_body = self.all_data['comment_body']
                train_data_doc2vec = submission_body.append(comments_body, ignore_index=True)
                train_data_doc2vec = pd.Series(train_data_doc2vec.unique())
                self.doc2vec_model = Doc2Vec(fname='', linux=False, use_file=False, data=train_data_doc2vec,
                                             vector_size=self.doc2vec_vector_size)
                joblib.dump(self.doc2vec_model, doc2vec_fitted_model_file_path)
                print('{}: finish fitting doc2vec model'.format(time.asctime(time.localtime(time.time()))))
                logging.info('finish fitting doc2vec model')

            else:
                print('{}: Loading fitted doc2vec model'.format(time.asctime(time.localtime(time.time()))))
                logging.info('Loading fitted doc2vec model')
                self.doc2vec_model = joblib.load(doc2vec_fitted_model_file_path)

        return

    def create_branch_submission_dict(self):
        """
        This function create branches features
        :return:
        """
        print(time.asctime(time.localtime(time.time())), ': Start branch features creation')
        logging.info('Start branch features creation')

        for index, branch_id in self.branch_ids.iteritems():
            if index % 1000 == 0:
                print(time.asctime(time.localtime(time.time())), ': Start branch id', branch_id,
                      'with branch index', index)
                logging.info('Start branch id {} with branch index {}'.format(branch_id, index))
            branch = self.all_branches.loc[self.all_branches.branch_id == branch_id]
            submission_id = branch.submission_id.values[0]
            branch_features = pd.Series(index=self.branch_submission_dict_features)
            branch_features.loc['branch_length'] = branch.branch_length.values[0]
            branch_first_comment = branch.branch_first_comment.values[0]
            branch_last_comment = branch.branch_last_comment.values[0]
            # all branches that their first comment was written before the last comment of the branch
            brothers_branches = self.all_branches.loc[(self.all_branches.branch_first_comment < branch_last_comment) &
                                                      (self.all_branches.submission_id == submission_id)]
            branch_features.loc['num_brother_branches_before_branch'] = brothers_branches.shape[0]
            branch_features.loc['mean_brothers_length'] = brothers_branches.branch_length.mean()
            # all deltas before the first comment of the branch
            submission_data = self.submission_comments_dict[submission_id]
            deltas_before_branch = submission_data.loc[(submission_data.comment_created_utc < branch_first_comment) &
                                                       (submission_data.delta == 1)]
            branch_features.loc['num_deltas_in_submission_before_branch'] = deltas_before_branch.shape[0]

            self.branch_submission_dict[branch_id] = [submission_id, np.array(branch_features)]

        return

    def create_branch_deltas_data_dict(self):
        """
        This function create delta features
        :return:
        """
        print(time.asctime(time.localtime(time.time())), ': Start branch deltas features creation')
        logging.info('Start branch deltas features creation')

        for index, branch_id in self.branch_ids.iteritems():
            if index % 1000 == 0:
                print(time.asctime(time.localtime(time.time())), ': Start branch id', branch_id,
                      'with branch index', index)
                logging.info('Start branch id {} with branch index {}'.format(branch_id, index))
            branch = self.all_branches.loc[self.all_branches.branch_id == branch_id]
            is_delta_in_branch = int(bool(branch.num_delta.values))  # 1 if there is deltas and 0 otherwise
            number_of_deltas_in_branch = branch.num_delta.values[0]
            deltas_comments_location_in_branch = list(self.data.loc[(self.data.branch_id == branch.branch_id.values[0]) &
                                                                    (self.data.delta == 1)]['comment_real_depth'])

            self.branch_deltas_data_dict[branch_id] = [is_delta_in_branch, number_of_deltas_in_branch,
                                                       deltas_comments_location_in_branch]

        return

    def create_submission_submitter_features(self):
        """
        This function create features of the submitter and submission that are not depend on time of the comments
        :return:
        """
        print(time.asctime(time.localtime(time.time())), ': Start submissions features creation')
        logging.info('Start submissions features creation')

        # Features calculated for all the data frame:
        self.all_submissions['submission_len'] = self.all_submissions['submission_body'].str.len()
        self.all_submissions['title_len'] = self.all_submissions['submission_title'].str.len()

        for sub_index, submission in self.all_submissions.iterrows():
            if sub_index % 100 == 0:
                print(time.asctime(time.localtime(time.time())), ': Start submission id', submission.submission_id,
                      'with submission index', sub_index)
                logging.info('Start submission id {} with submission index {}'.
                             format(submission.submission_id, sub_index))
            submission_features = pd.DataFrame(columns=self.submission_features_columns)
            submitter_features = pd.DataFrame(columns=self.submitter_features_columns)

            # create submission features
            submission_features.loc[0, 'submission_len'] = submission['submission_len']
            submission_features.loc[0, 'title_len'] = submission['title_len']

            # sentiment analysis for the submission body
            sub_sentiment_list = sentiment_analysis(submission['submission_body'])
            submission_features.loc[0, 'nltk_sub_sen_pos'], submission_features.loc[0, 'nltk_sub_sen_neg'], \
                submission_features.loc[0, 'nltk_sub_sen_neutral'] = \
                sub_sentiment_list[0], sub_sentiment_list[1], sub_sentiment_list[2]
            # for the title:
            title_sentiment_list = sentiment_analysis(submission['submission_title'])
            submission_features.loc[0, 'nltk_title_sen_pos'], submission_features.loc[0, 'nltk_title_sen_neg'], \
                submission_features.loc[0, 'nltk_title_sen_neutral'] = \
                title_sentiment_list[0], title_sentiment_list[1], title_sentiment_list[2]

            # time
            submission_features.loc[0, 'time_until_first_comment'], _ =\
                self.time_to_first_comment(submission.submission_id, submission.submission_created_utc, 0)

            # create submitter features
            submitter_features.loc[0, 'submitter_number_submission'] = \
                self.number_of_message(submission.submission_author, submission.submission_created_utc, 'submission')
            submitter_features.loc[0, 'submitter_number_comment'] = \
                self.number_of_message(submission.submission_author, submission.submission_created_utc, 'comment')
            submitter_features.loc[0, 'submitter_seniority_days'] =\
                self.calculate_user_seniority(submission.submission_author)
            comments_in_submission, number_of_comments_in_tree_from_submitter = \
                self.comments_in_submission(submission.submission_id, submission.submission_author)
            submission_features.loc[0, 'comments_in_submission'] = comments_in_submission
            if comments_in_submission == 0:
                print('no comments in submission', submission.submission_id)
                logging.info('no comments in submission', submission.submission_id)
                submitter_features.loc[0, 'comments_total_ratio'] = 0
            else:
                submitter_features.loc[0, 'comments_total_ratio'] =\
                    number_of_comments_in_tree_from_submitter / comments_in_submission
            submitter_features.loc[0, 'number_of_comments_in_tree_from_submitter'] =\
                number_of_comments_in_tree_from_submitter

            embedded_submission_text = self.doc2vec_model.infer_doc_vector(submission.submission_title_and_body)

            self.submission_data_dict[submission.submission_id] = [embedded_submission_text,
                                                                   np.array(submission_features)[0].astype(float),
                                                                   np.array(submitter_features)[0].astype(float)]

        return

    def create_branch_comments_text(self):
        """
        This function create the branch_comments_raw_text_df - loop over branches sorted by their length and append
        the comments body
        :return:
        """

        print(time.asctime(time.localtime(time.time())), ': Start comments text creation')
        logging.info('Start comments text creation')

        for index, branch_id in self.branch_ids.iteritems():
            if index % 1000 == 0:
                print(time.asctime(time.localtime(time.time())), ': Start branch_id', branch_id,
                      'with branch index', index)
                logging.info('Start branch_id {} with branch index {}'.format(branch_id, index))
            branch_comments_body = self.branch_comments_dict[branch_id][['comment_body']]
            branch_comments_body = branch_comments_body.assign(embedded_comment_text='')
            for inner_index, comment in branch_comments_body.iterrows():
                branch_comments_body.loc[inner_index, 'embedded_comment_text'] = \
                    self.doc2vec_model.infer_doc_vector(comment['comment_body'])
            branch_comments_body = branch_comments_body['embedded_comment_text']
            if branch_comments_body.shape[0] < self.max_branch_length:
                append_zero = pd.Series(np.zeros(
                    shape=(self.max_branch_length - branch_comments_body.shape[0], self.doc2vec_vector_size)).tolist())
                branch_comments_body = pd.concat([branch_comments_body, append_zero], ignore_index=True)
            else:
                branch_comments_body = branch_comments_body.reset_index()['embedded_comment_text']
            # branch_comments_body.name = index

            self.branch_comments_embedded_text_df[branch_id] = branch_comments_body

            # self.branch_comments_embedded_text_df = self.branch_comments_embedded_text_df.append(branch_comments_body)

        self.branch_comments_embedded_text_df = pd.DataFrame.from_records(
            list(self.branch_comments_embedded_text_df.values()), index=list(self.branch_comments_embedded_text_df.keys()))
        print(time.asctime(time.localtime(time.time())), ': Finish comments text creation')
        logging.info('Finish comments text creation')

        return

    def create_branch_comments_features_df(self):
        """
        This function create the branch_comments_features_df and branch_comments_user_profiles_df for all commments -
        go over all branches and all comments in the branch and create its features
        :return:
        """

        print(time.asctime(time.localtime(time.time())), ': Start comments and commenters features creation')
        logging.info('Start comments and commenters features creation')

        # Get topic model result
        topic_model_result = self.topic_model()

        # # get submission_num_comments
        # comment_submission = self.data[['comment_id', 'submission_id']]
        # comment_submission_groupby = comment_submission.groupby(by='submission_id').count()
        # comment_submission_groupby['submission_id'] = comment_submission_groupby.index
        # comment_submission_groupby.columns = ['submission_num_comments', 'submission_id']
        #
        # self.data = self.data.merge(comment_submission_groupby, on='submission_id')

        # get each comment once
        comments = self.data.drop_duplicates(subset='comment_id').reset_index(drop=True)
        print(time.asctime(time.localtime(time.time())), ': Number of comments to infer', comments.shape[0])
        logging.info('Number of comments to infer {}'.format(comments.shape[0]))

        all_comments_features = dict()
        all_comments_user_features = dict()
        # create features for each comment - not matter the branch
        for comment_index, comment in comments.iterrows():
            if comment_index % 1000 == 0:
                print(time.asctime(time.localtime(time.time())), ': Start comment_id', comment['comment_id'],
                      'with comment index', comment_index)
                logging.info('Start comment_id {} with comment index {}'.format(comment['comment_id'], comment_index))
            comment_features = pd.Series(index=self.comment_features_columns)
            comment_user_features = pd.Series(index=self.comments_user_features_columns)

            # get comment info
            comment_author = copy.deepcopy(comment['comment_author'])
            comment_time = copy.deepcopy(comment['comment_created_utc'])
            submission_time = copy.deepcopy(comment['submission_created_utc'])
            submission_id = copy.deepcopy(comment['submission_id'])
            comment_body = copy.deepcopy(comment['comment_body'])
            submission_body = copy.deepcopy(comment['submission_body'])
            submission_author = comment['submission_author']

            # add topic model features and comment_len and comment_real_depth

            topic_model_comment = pd.Series(topic_model_result.loc[
                                                topic_model_result.comment_id == comment['comment_id'],
                                                topic_model_result.columns != 'comment_id'].values[0])
            comment_features = comment_features.append(topic_model_comment)
            comment_features.loc['comment_len'] = comment['comment_len']
            comment_features.loc['comment_real_depth'] = comment['comment_real_depth']
            comment_features.loc['number_of_branches'] = comment['number_of_branches']

            # number of times another subreddit or a url was mentioned
            comment_features.loc['num_mentioned_subreddit'] = mentioned_another_subreddit(comment_body)
            comment_features.loc['num_mentioned_url'] = contain_url(comment_body)

            # treatment:
            comment_features.loc['is_quote'] = self.loop_over_comment_for_quote(comment, comment_body)
            # Get the time between the submission and the comment time and the ration between the first comment:
            time_to_comment = comment['time_between']
            time_between_messages_hour = math.floor(time_to_comment / 3600.0)
            time_between_messages_min = math.floor(
                (time_to_comment - 3600 * time_between_messages_hour) / 60.0) / 100.0
            comment_features.loc['time_between_sub_com'] = time_between_messages_hour + time_between_messages_min
            time_until_first_comment, time_between_comment_first_comment = \
                self.time_to_first_comment(submission_id, submission_time, comment_time)
            if time_to_comment > 0:
                comment_features.loc['time_ratio_first_comment'] = time_until_first_comment / time_to_comment
            else:
                comment_features.loc['time_ratio_first_comment'] = 0

            comment_features.loc['time_between_comment_first_comment'] = time_between_comment_first_comment

            # Sentiment analysis:
            # for the comment:
            comment_sentiment_list = sentiment_analysis(comment_body)
            comment_features.loc['nltk_com_sen_pos'], comment_features.loc['nltk_com_sen_neg'], \
            comment_features.loc['nltk_com_sen_neutral'] = \
                comment_sentiment_list[0], comment_sentiment_list[1], comment_sentiment_list[2]
            # for the submission:
            sub_sentiment_list = sentiment_analysis(submission_body)
            # cosine similarity between submission's sentiment vector and comment sentiment vector:
            sentiment_sub = np.array(sub_sentiment_list).reshape(1, -1)
            sentiment_com = np.array(comment_sentiment_list).reshape(1, -1)
            comment_features.loc['nltk_sim_sen'] = cosine_similarity(sentiment_sub, sentiment_com)[0][0]

            # percent of adjective in the comment:
            comment_features.loc['percent_adj'] = percent_of_adj(comment_body, comment['comment_id'])

            # Get comment author features:
            comment_user_features.loc['commenter_number_submission'] = \
                self.number_of_message(comment_author, comment_time, 'submission')
            comment_user_features.loc['commenter_number_comment'] = \
                self.number_of_message(comment_author, comment_time, 'comment')
            comment_user_features.loc['commenter_seniority_days'] = self.calculate_user_seniority(comment_author)
            comment_user_features.loc['is_first_comment_in_tree'], \
            comment_user_features.loc['number_of_comments_in_submission_by_comment_user'], _, _, \
            submission_num_comments = \
                self.comment_in_tree(comment_author, comment_time, submission_id)
            comment_features.loc['submission_num_comments'] = submission_num_comments

            # Get the numbers of comments by the submitter
            _, _, number_of_respond_by_submitter, number_of_respond_by_submitter_total, _ = \
                self.comment_in_tree(submission_author, comment_time, submission_id, comment_author, True)
            comment_user_features.loc['number_of_respond_by_submitter_to_commenter'] = \
                number_of_respond_by_submitter
            # Ratio of comments number:
            if submission_num_comments == 0:
                comment_user_features.loc['respond_to_comment_user_all_ratio'] = 0
            else:
                comment_user_features.loc['respond_to_comment_user_all_ratio'] = \
                    number_of_respond_by_submitter / submission_num_comments
            if number_of_respond_by_submitter_total == 0:
                comment_user_features.loc['respond_to_comment_user_responses_ratio'] = 0
            else:
                comment_user_features.loc['respond_to_comment_user_responses_ratio'] = \
                    number_of_respond_by_submitter / number_of_respond_by_submitter_total

            # sim feature:
            comment_user_features.loc['submmiter_commenter_tfidf_cos_sim'] = \
                self.calc_tf_idf_cos(comment_time, comment_author, submission_author)

            # append comments features to all comments features
            all_comments_features[comment['comment_id']] = comment_features
            all_comments_user_features[comment['comment_id']] = comment_user_features

        # insert comment and comment user features to features DF
        print(time.asctime(time.localtime(time.time())),
              ': Start insert comment and comment user features to features DF')
        logging.info('insert comment and comment user features to features DF')
        for branch_index, branch_id in self.branch_ids.iteritems():
            branch_comments_features = pd.Series(np.zeros(
                shape=(1, self.comment_features_columns_len)).tolist() * self.max_branch_length)
            branch_comments_user_features = pd.Series(np.zeros(
                shape=(1, len(self.comments_user_features_columns))).tolist() * self.max_branch_length)
            if branch_index % 1000 == 0:
                print(time.asctime(time.localtime(time.time())), ': Start branch_id', branch_id,
                      'with branch index', branch_index)
                logging.info('Start branch_id {} with branch index {}'.format(branch_id, branch_index))
            branch_comments = self.branch_comments_dict[branch_id]
            branch_comments = \
                branch_comments.sort_values(by='comment_real_depth', ascending=True).reset_index(drop=True)

            for comment_index, comment in branch_comments.iterrows():
                comment_features = all_comments_features[comment['comment_id']]
                comment_user_features = all_comments_user_features[comment['comment_id']]

                branch_comments_features.loc[comment_index] = np.array(comment_features).astype(float)
                branch_comments_user_features.loc[comment_index] = np.array(comment_user_features).astype(float)

            self.branch_comments_features_df[branch_id] = branch_comments_features
            self.branch_comments_user_profiles_df[branch_id] = branch_comments_user_features

            # self.branch_comments_features_df.loc[branch_index, comment_index] =\
            #     np.array(comment_features.loc[:, comment_features.columns != 'comment_id'])[0].astype(float)
            # self.branch_comments_user_profiles_df.loc[branch_index, comment_index] =\
            #     np.array(comment_user_features.loc[:, comment_user_features.columns != 'comment_id'])[0].\
            #         astype(float)

        # self.branch_comments_features_df = self.branch_comments_features_df.fillna(
        #     np.zeros(shape=len(self.comment_features_columns)))
        # self.branch_comments_user_profiles_df = self.branch_comments_user_profiles_df.fillna(
        #     np.zeros(shape=len(self.comments_user_features_columns)))

        self.branch_comments_features_df = pd.DataFrame.from_records(
            list(self.branch_comments_features_df.values()), index=list(self.branch_comments_features_df.keys()))
        self.branch_comments_user_profiles_df = pd.DataFrame.from_records(
            list(self.branch_comments_user_profiles_df.values()), index=list(self.branch_comments_user_profiles_df.keys()))

        return

    def create_all_features(self, features_dir_path=features_directory):
        """
        This function first create features that are calculated for all the data frame and then features for each unit
        before each function, check if the file is already exists, if no, call the function and save the file
        :param features_dir_path: in which directory to save the features
        :return:
        """

        # create branch_deltas_data_dict
        file_path = os.path.join(features_dir_path, 'branch_deltas_data_dict_' + self.data_file_name + '.pickle')
        if not os.path.isfile(file_path):
            self.create_branch_deltas_data_dict()
            with open(file_path, 'wb') as handle:
                pickle.dump(self.branch_deltas_data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # create branch comments text data frame
        file_path = os.path.join(features_dir_path, 'branch_comments_embedded_text_df_' + self.data_file_name + '.pkl')
        if not os.path.isfile(file_path):
            self.create_branch_comments_text()
            # save_as_pickled_object(self.branch_comments_embedded_text_df, file_path + '.pkl')
            joblib.dump(self.branch_comments_embedded_text_df, file_path)

        # create branch_comments_features_df and branch_comments_user_profiles_df
        file_path1 = os.path.join(features_dir_path, 'branch_comments_features_df_' + self.data_file_name + '.pkl')
        file_path2 = os.path.join(features_dir_path, 'branch_comments_user_profiles_df_' +
                                  self.data_file_name + '.pkl')
        if not os.path.isfile(file_path1) or not os.path.isfile(file_path2):
            self.create_branch_comments_features_df()
            # joblib.dump(self.branch_comments_features_df, file_path1 + '.compressed', compress=True)
            # joblib.dump(self.branch_comments_user_profiles_df, file_path2 + '.compressed', compress=True)
            joblib.dump(self.branch_comments_features_df, file_path1)
            joblib.dump(self.branch_comments_user_profiles_df, file_path2)

        # create submission data dict
        file_path = os.path.join(features_dir_path, 'submission_data_dict_' + self.data_file_name + '.pickle')
        if not os.path.isfile(file_path):
            self.create_submission_submitter_features()
            with open(file_path, 'wb') as handle:
                pickle.dump(self.submission_data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # create branch features
        file_path = os.path.join(features_dir_path, 'branch_submission_dict_' + self.data_file_name + '.pickle')
        if not os.path.isfile(file_path):
            self.create_branch_submission_dict()
            with open(file_path, 'wb') as handle:
                pickle.dump(self.branch_submission_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # save branches length
        with open(os.path.join(features_dir_path, 'branches_lengths_list_' + self.data_file_name + '.txt'), 'wb')\
                as handle:
            pickle.dump(self.branches_lengths_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return

    def number_of_message(self, user, comment_time, messages_type):
        """
        Get the number of messages (submissions and comments) the user posted we have in the data
        :param str user: the user name we want to check
        :param int comment_time: the time the comment in the unit was posted (time t)
        :param str messages_type: submission / comment - what we want to check
        :return: the number of messages of the messages_type
        :rtype int
        """
        if messages_type == 'comment':
            relevant_data = self.all_data.loc[(self.all_data[messages_type + '_author'] == user)
                                              & (self.all_data[messages_type + '_created_utc'] < comment_time)]
        else:
            relevant_data =\
                self.all_data_set_submissions.loc[
                    (self.all_data_set_submissions[messages_type + '_author'] == user)
                    & (self.all_data_set_submissions[messages_type + '_created_utc'] < comment_time)]
        number_of_posts = relevant_data.shape[0]

        return number_of_posts

    def comments_in_submission(self, submission_id, submitter):
        """
        Number of comments in the submission
        :param int submission_id: the submission id
        :param str submitter: the user that wrote the submission
        :return:
        """
        submission_data = self.submission_comments_dict[submission_id]
        comments_by_submitter = submission_data[submission_data['comment_author'] == submitter]

        return submission_data.shape[0], comments_by_submitter.shape[0]

    def comment_in_tree(self, user, comment_time, submission_id, comment_user=None,
                        submitter_respond_to_comment_user=False):
        """
        Check if this is the first comment the comment author posted for this submission
        :param str user: the user name we want to check (either submitter or comment user)
        :param int comment_time: the time the comment in the unit was posted (time t), if =0: this is the submission
        :param int submission_id: the submission id
        :param str comment_user: the comment user name of this unit
        :param bool submitter_respond_to_comment_user: whether we check the submitter_respond_to_comment_user or not
        :return: int is_first_comment_in_tree: 1 - if this is the first time, 0 - otherwise
                int number_of_comments_in_tree: number of comments he wrote in the submission tree until time t
                int number_of_respond_by_submitter: the number of responds by the submitter to the comment user
                int number_of_respond_by_submitter_total: the number of responds by the submitter in total
                int number_comments_in_submission: the number of all comments in submission until time t
        """

        submission_data = self.submission_comments_dict[submission_id]
        if comment_time == 0:  # get the last time of comment in submission
            comment_time = submission_data.comment_created_utc.max()
        # if user is the submission user - these are the comments he wrote
        all_comments_user_in_tree = submission_data.loc[(submission_data['comment_author'] == user)
                                                        & (submission_data['comment_created_utc'] < comment_time)]
        comment_submission_before_comment = submission_data[submission_data['comment_created_utc'] < comment_time]
        if all_comments_user_in_tree.empty:
            number_of_comments_in_tree = 0
            is_first_comment_in_tree = 1
            # if there are no comments before comment_time - if this is the submitter, no need to check the
            # number_of_respond_by_submitter and number_of_respond_by_submitter_total - they will be 0
            number_of_respond_by_submitter = 0
            number_of_respond_by_submitter_total = 0
            number_comments_in_submission =\
                comment_submission_before_comment.shape[0] if not comment_submission_before_comment.empty else 0
            return is_first_comment_in_tree, number_of_comments_in_tree, number_of_respond_by_submitter,\
                number_of_respond_by_submitter_total, number_comments_in_submission
        else:  # if there are comments in before comment_time from this user
            number_of_comments_in_tree = all_comments_user_in_tree.shape[0]
            is_first_comment_in_tree = 0
            number_comments_in_submission = \
                comment_submission_before_comment.shape[0]

            if not submitter_respond_to_comment_user:  # if this the comment user
                return is_first_comment_in_tree, number_of_comments_in_tree, 0, 0, number_comments_in_submission
            else:  # submitter = user
                # the parent ids of all the comments that were written by the submitter in this submission until time t
                parent_id_list = list(all_comments_user_in_tree['parent_id'])
                # take all comments in this submission that were written by the comment user and are the parents of the
                # submitter's comments - i.e the submitter respond to the comment user
                parent_by_the_comment_author = submission_data[(submission_data['comment_id'].isin(parent_id_list))
                                                               & (submission_data['comment_author'] == comment_user)
                                                               & (submission_data['comment_created_utc'] < comment_time)]

                # number of responses by submitter : parent_id != submission_id -
                # the submitter wrote a comment to someone - respond
                respond_by_submitter_total =\
                    all_comments_user_in_tree.loc[all_comments_user_in_tree['parent_id'] != submission_id]

                number_of_respond_by_submitter = parent_by_the_comment_author.shape[0]
                number_of_respond_by_submitter_total = respond_by_submitter_total.shape[0]

                return is_first_comment_in_tree, number_of_comments_in_tree, number_of_respond_by_submitter,\
                    number_of_respond_by_submitter_total, number_comments_in_submission

    def time_to_first_comment(self, submission_id, submission_created_time, comment_created_time):
        """
        Calculate the time between the submission and the first comment
        :param int submission_id: the submission id
        :param int submission_created_time: the utc time of the submission
        :param int comment_created_time: the utc time of the comment
        :return: int the seconds between the submission and the first comment in its tree
        """

        all_submission_comments = self.submission_comments_dict[submission_id]
        time_of_first_comment = all_submission_comments['comment_created_utc'].min()
        time_until_first_comment = time_of_first_comment - submission_created_time
        time_between_comment_first_comment = comment_created_time - time_of_first_comment

        return time_until_first_comment, time_between_comment_first_comment

    def calculate_user_seniority(self, user):
        """
        Calculate the user seniority in change my view subreddit in days
        :param str user: the user name
        :return: int the number of days since the first post of the user (submissions and comments)
        """
        user_all_comments = self.all_data.loc[self.all_data['comment_author'] == user]
        user_all_submissions = self.all_data_set_submissions.loc[
            self.all_data_set_submissions['submission_author'] == user]
        first_comment_time = user_all_comments.comment_created_utc.min()
        first_submission_time = user_all_submissions.submission_created_utc.min()
        if not user_all_comments.empty:  # the user has not write any comment - so the min will be nan
            first_post_time = min(first_comment_time, first_submission_time)
        else:
            if not user_all_submissions.empty:
                first_post_time = min(first_submission_time, first_comment_time)
            else:
                print('no comments and submission for user: {}'.format(user))
                logging.info('no comments and submission for user: {}'.format(user))
                return 0
        tz = pytz.timezone('GMT')  # America/New_York
        date_comment = datetime.fromtimestamp(first_post_time, tz)
        utc_now = int(time.time())
        date_now = datetime.fromtimestamp(utc_now, tz)
        time_between = date_now - date_comment

        return time_between.days

    def loop_over_comment_for_quote(self, comment, comment_body):
        """
        Go over the comment and check if there is quote in each part
        :param pandas series comment: a series with all the comment's information
        :param str comment_body: the comment's body
        :return: 0 if there is no quote in this part of comment, 1 if there is, -1 if we don't want this unit
        """
        is_quote = 0
        if '>' in comment_body:  # this is the sign for a quote (|) in the comment
            while '>' in comment_body and not is_quote:
                quote_index = comment_body.find('>')
                comment_body = comment_body[quote_index + 1:]
                is_quote = self.check_quote(comment, comment_body)
        else:  # there is not quote at all
            is_quote = 0

        return is_quote

    def check_quote(self, comment, comment_body):
        """
        Check if there is a quote in the comment.
        Check if it is a quote of a comment of the submitter or the submission itself
        :param pandas series comment: a series with all the comment's information
        :param str comment_body: the comment's body
        :return: 0 if there is no quote in this part of comment, 1 if there is, -1 if we don't want this unit
        """

        no_parent = False
        quote = copy.deepcopy(comment_body)
        nn_index = quote.find('\\n')
        n_index = quote.find('\n')
        if nn_index == -1:  # there is no \\n
            quote = quote
        elif n_index != -1:  # there is \n
            quote = quote[: n_index - 1]
        else:
            quote = quote[: nn_index - 1]  # take the quote: after the > and until the first \n
        # parse the parent id
        parent_id = comment['parent_id']

        # if the parent is the submission - take the submission body
        if parent_id == comment['submission_id']:
            parent_body = comment['submission_body']
            parent_author = comment['submission_author']
        else:  # if not - get the parent
            parent = self.data.loc[self.data['comment_id'] == parent_id]
            if parent.empty:  # if we don't have the parent as comment in the data
                parent_body = ''
                parent_author = ''
                no_parent = True
            else:
                parent = pd.Series(parent.iloc[0])
                parent_body = parent['comment_body']
                parent_author = parent['comment_author']
        submission_author = comment['submission_author']
        submission_body = comment['submission_body']
        submission_title = comment['submission_title']

        if submission_author == parent_author:  # check if the parent author is the submitter
            # if he quote the submission or the parent
            if (quote in parent_body) or (quote in submission_body) or (quote in submission_title):
                return 1
            else:  # he didn't quote the submitter
                return 0
        else:  # if the parent author is not the submitter
            if (quote in submission_body) or (quote in submission_title):  # we only care of he quote the submission:
                return 1
            else:
                if no_parent:
                    # if there is no parent and he didn't quote the submission, we can't know if he quote the parent
                    # - so we don't need to use it
                    return -1
                else:
                    return 0

    def create_data_dictionary(self):
        """
        Clean data and create dictionary for topic model
        :return: dictionary
        """
        # Clean the data
        print('{}: Clean the data'.format((time.asctime(time.localtime(time.time())))))
        logging.info('Clean the data')

        data_clean = {row['comment_id']: clean(row['comment_body'], row['comment_id']).split()
                      for index, row in self.data.iterrows()}

        # Creating the term dictionary of our corpus, where every unique term is assigned an index.
        print('{}: Create the dictionary'.format((time.asctime(time.localtime(time.time())))))
        logging.info('Create the dictionary')

        dictionary = gensim.corpora.Dictionary(data_clean.values())

        # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
        print('{}: Create units term matrix'.format((time.asctime(time.localtime(time.time())))))
        logging.info('Create units term matrix')

        data_term_matrix = {index: dictionary.doc2bow(doc) for index, doc in data_clean.items()}

        return dictionary, data_term_matrix

    def topic_model(self):
        """
        Calculate the topic model for all the units, the probability that the comment has each of the topics
        :return: pandas DF[number_of_units, number_of_topics] - the probability for each comment and topic
        """
        if self.is_train:  # if train data - we already created the train_data_term_matrix for the model creation
            data_term_matrix = self.train_data_term_matrix

        else:
            # Clean the data
            print('{}: Clean the {}'.format((time.asctime(time.localtime(time.time()))), self.data_file_name))
            logging.info('Clean the {}'.format(self.data_file_name))

            data_clean = {row['comment_id']: clean(row['comment_body'], row['comment_id']).split()
                          for index, row in self.data.iterrows()}

            # Creating the term dictionary of our corpus, where every unique term is assigned an index.
            print('{}: Create the dictionary for {}'.format((time.asctime(time.localtime(time.time()))),
                                                            self.data_file_name))
            logging.info('Create the dictionary for {}'.format(self.data_file_name))
            dictionary = gensim.corpora.Dictionary(data_clean.values())

            # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
            print('{}: Create data term matrix for {}'.format((time.asctime(time.localtime(time.time()))),
                                                              self.data_file_name))
            logging.info('Create data term matrix for {}'.format(self.data_file_name))
            data_term_matrix = {index: dictionary.doc2bow(doc) for index, doc in data_clean.items()}

        # Get topics for the data
        print('{}: Predict topics for {}'.format((time.asctime(time.localtime(time.time()))), self.data_file_name))
        logging.info('Predict topics for {}'.format(self.data_file_name))

        result = self.lda_model.transform(list(data_term_matrix.values()))

        print('{}: Create final topic model for {}'.format((time.asctime(time.localtime(time.time()))),
                                                           self.data_file_name))
        logging.info('Create final topic model for {}'.format(self.data_file_name))
        comment_ids_df = pd.DataFrame(list(data_term_matrix.keys()), columns=['comment_id'])
        result_columns = [i for i in range(self.number_of_topics)]
        topic_model_result_df = pd.DataFrame(result, columns=result_columns)

        print('{}: Save final topic model for {}'.format((time.asctime(time.localtime(time.time()))),
                                                         self.data_file_name))
        logging.info('Save final topic model for {}'.format(self.data_file_name))
        topic_model_final_result = pd.concat([comment_ids_df, topic_model_result_df], axis=1)

        return topic_model_final_result

    def data_pre_process(self):
        """
        This function do some pre process to the submission data
        :return:
        """

        print('{}: begin data pre process'.format(time.asctime(time.localtime(time.time()))))
        logging.info('begin data pre process')

        # get the relevant submission for the data
        submission_list = list(self.data['submission_id'].unique())
        self.all_submissions = copy.deepcopy(
            self.all_submissions_total.loc[self.all_submissions_total['submission_id'].isin(submission_list)])

        all_data_submission_list = list(self.all_data['submission_id'].unique())
        self.all_data_set_submissions =\
            copy.deepcopy(self.all_submissions_total.loc[self.all_submissions_total['submission_id'].isin(
                all_data_submission_list)])

        branches_list = list(self.data['branch_id'].unique())
        self.all_branches = copy.deepcopy(
            self.branch_numbers_df.loc[self.branch_numbers_df.branch_id.isin(branches_list)])

        # remove bot text
        self.all_submissions["submission_body"] = self.all_submissions["submission_body"].str.partition(
            "Hello, users of CMV! This is a footnote from your moderators")[0]

        # concat submissions text and title
        self.all_submissions["submission_title_and_body"] =\
            self.all_submissions["submission_title"] + self.all_submissions["submission_body"]
        self.all_data_set_submissions["submission_title_and_body"] =\
            self.all_data_set_submissions["submission_title"] + self.all_data_set_submissions["submission_body"]

        print('{}: finish data pre process'.format(time.asctime(time.localtime(time.time()))))
        logging.info('finish data pre process')

        return

    def concat_df_rows(self, comment_created_utc, author, is_submission=False):
        """
        This function concat all the comments or all the submissions of a particular user
        :param int comment_created_utc: the time the comment was posted
        :param str author: the name of the author
        :param bool is_submission: whether we want to concat submissions of comments
        :return: the concatenated text
        """
        if is_submission:
            text = self.all_data_set_submissions.loc[
                (self.all_data_set_submissions['submission_created_utc'] <= comment_created_utc) &
                (self.all_data_set_submissions['submission_author'] == author)]["submission_title_and_body"]
            text_cat = text.str.cat(sep=' ')
            return text_cat

        text = self.all_data.loc[(self.all_data['comment_created_utc'] <= comment_created_utc) &
                                 (self.all_data['comment_author'] == author)]["comment_body"]
        text_cat = text.str.cat(sep=' ')

        return text_cat

    def create_vocab(self):
        """
        This function create a vocabulary - all the submissions and comments body in the data.
        Create the vocab over the all data set and not only the split
        :return:
        """

        # get all comments for vocab
        vocab_c = self.all_data["comment_body"]
        vocab_s = self.all_data_set_submissions["submission_title_and_body"]

        # join two strings of comments and submissions
        vocab_df = pd.concat([vocab_c, vocab_s])
        vocab_df = vocab_df.drop_duplicates()
        vocab_df = vocab_df.reset_index()
        vocab_df = vocab_df[0]
        vocab_df = vocab_df.astype(str)
        return vocab_df

    def calc_tf_idf_cos(self, comment_created_utc, comment_author, submission_author):
        """
        This function calculate the TFIDF similarity between the submitter and commenter text
        before the comment was written for all the data, not only the split
        :param int comment_created_utc: the time the comment was posted
        :param str comment_author: the name of the comment author
        :param str submission_author: the name of the submission author
        :return: the cosine similarity between the commenter and the submitter text in the data
        """

        # all text of commenter until comment time
        text_commenter = self.concat_df_rows(comment_created_utc, comment_author)
        text_commenter_submission = self.concat_df_rows(comment_created_utc, comment_author, True)
        text_commenter += text_commenter_submission

        # all text of submitter until comment time
        text_submitter = self.concat_df_rows(comment_created_utc, submission_author)
        text_submitter_submission = self.concat_df_rows(comment_created_utc, submission_author, True)
        text_submitter += text_submitter_submission

        text = [text_submitter, text_commenter]

        tfidf_vec_transformed = self.tfidf_vec_fitted.transform(text)

        similarity = cosine_similarity(tfidf_vec_transformed[0:], tfidf_vec_transformed[1:])

        return similarity[0][0]


def sentiment_analysis(text):
    """
    This function calculate the sentiment of a text. It is calculate the probability of the text to be negative,
    positive or neutral
    :param str text: the text we calculate its sentiments
    :return: list: pos_prob, neg_prob, neutral_prob
    """
    result = sid.polarity_scores(text)
    neg_prob = result['neg']
    neutral_prob = result['neu']
    pos_prob = result['pos']
    return [pos_prob, neg_prob, neutral_prob]


def get_POS(text):
    """
    This function find the POS of each word in text.
    :param str text: the text we want to find its POS
    :return: list(tuple) - a list of the words- for each a tuple: (word, POS)
    """
    text_parsed = nk.word_tokenize(text)
    words_pos = nk.pos_tag(text_parsed)

    return words_pos


def percent_of_adj(text, comment_id):
    """
    This function calculate the % of the adjectives in the text
    :param str text: the text we want to calculate its %
    :return: float: number_adj_pos/number_all_pos in the text
    """
    pos_text = get_POS(text)
    pos_df = pd.DataFrame(pos_text, columns=['word', 'POS'])
    number_all_pos = pos_df.shape[0]
    all_pos = pos_df['POS']
    freq = nk.FreqDist(all_pos)
    number_adj_pos = freq['JJ'] + freq['JJS'] + freq['JJR']
    percent_of_adj_pos = number_adj_pos/number_all_pos

    return percent_of_adj_pos


def clean(text, comment_id):
    """
    This function clean a text from stop words and punctuations and them lemmatize the words
    :param str text: the text we want to clean
    :return: str normalized: the cleaned text
    """
    if type(text) != str:
        print(text, comment_id)
    text = text.lstrip('b').strip('"').strip("'").strip(">")
    stop_free = " ".join([i for i in text.lower().split() if i not in stop])
    punc_free = "".join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized


def mentioned_another_subreddit(comment_body: str) -> int:
    """
    The number of other subreddits that mentioned the comment
    :param comment_body: the comment to check
    :return: the number of times another subreddit was mentioned in the comment
    """
    reference_subreddit = comment_body[comment_body.find('r'):].split('/')  # split the body from /r/
    number_of_r = 0
    comment_list_len = len(reference_subreddit)
    for i in range(0, comment_list_len):
        # consider as reference_subreddit if the subreddit that is mentioned is not changemyview
        if reference_subreddit[i] == 'r' and i + 1 < comment_list_len and reference_subreddit[i + 1] != 'changemyview':
            number_of_r += 1
    return number_of_r


def contain_url(comment_body: str) -> int:
    """
    The number of urls that are in the comment
    :param comment_body: the comment to check
    :return: the number of times a url was mentioned in the comment
    """
    # findall() has been used with valid conditions for urls in string
    url = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+] |[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', comment_body)
    return len(url)


def split_data_and_run(data_type: str, create_features: CreateFeatures, split_number: int):
    print(f'{time.asctime(time.localtime(time.time()))}: Split {data_type} data to create features')
    logging.info('Split {} data to create features'.format(data_type))

    all_data = create_features.data
    data_submissions = all_data.submission_id.unique()
    print('number of submissions in train data:', data_submissions.shape[0])
    number_to_choose = round(data_submissions.shape[0] / split_number)
    last_group_size = data_submissions.shape[0] - number_to_choose * (split_number - 1)
    group_sizes = [number_to_choose] * (split_number - 1)
    group_sizes.append(last_group_size)
    for group_id, group_size in enumerate(group_sizes):
        if group_id == 0:
            prev_group_size = 0
        else:
            prev_group_size = sum(group_sizes[:group_id])
        submissions_to_choose = data_submissions[prev_group_size:prev_group_size + group_size]
        data_set = all_data.loc[all_data.submission_id.isin(submissions_to_choose)].copy()
        print('{}: Start loading {} data, group number {}'.
              format((time.asctime(time.localtime(time.time()))), data_type, group_id))
        logging.info('Start loading {} data, group number {}'.format(data_type, group_id))

        create_features.create_data(data_type + '_' + str(group_id), is_train=False, data=data_set)
        print('{}: Finish loading the data, start create features'.
              format((time.asctime(time.localtime(time.time())))))
        print('data sizes: {} data, group number {}: {}'.format(data_type, group_id, create_features.data.shape))
        logging.info('Finish loading the data, start create features')
        logging.info('data sizes: {} data, group number {}: {}'.format(data_type, group_id, create_features.data.shape))
        create_features.create_all_features()

    return


def not_parallel_main():
    data_to_create_features = ['train']
    log_file_name = os.path.join(log_directory, datetime.now().strftime(
                                    f'LogFile_create_features_delta_{data_to_create_features}_%d_%m_%Y_%H_%M_%S.log'))
    logging.basicConfig(filename=log_file_name,
                        level=logging.DEBUG,
                        format='%(asctime)s: %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        )
    topics_number = 15
    split_number = 10
    print('{}: Create object'.format((time.asctime(time.localtime(time.time())))))
    logging.info('Create object')
    create_features = CreateFeatures(topics_number)

    print('{}: Loading train data'.format((time.asctime(time.localtime(time.time())))))
    logging.info('Loading train data')
    create_features.create_data('train', is_train=True)

    print('{}: Finish loading the data'.format((time.asctime(time.localtime(time.time())))))
    print('data sizes: train data: {}'.format(create_features.data.shape))
    logging.info('Finish loading the data')
    logging.info('data sizes: train data: {}'.format(create_features.data.shape))

    if 'train' in data_to_create_features:
        split_data_and_run('train', create_features, split_number)
        print('{}: Finish creating train data features'.format((time.asctime(time.localtime(time.time())))))
        logging.info('Finish creating train data features')

    if 'test' in data_to_create_features:
        print('{}: Start loading test data'.format((time.asctime(time.localtime(time.time())))))
        logging.info('Start loading test data')
        create_features.create_data('test', is_train=False)

        split_data_and_run('test', create_features, split_number)

        print('{}: Finish creating test data features'.format((time.asctime(time.localtime(time.time())))))
        logging.info('Finish creating test data features')

    if 'val' in data_to_create_features:
        print('{}: Start loading val data'.format((time.asctime(time.localtime(time.time())))))
        logging.info('Start loading val data')
        create_features.create_data('val', is_train=False)

        split_data_and_run('val', create_features, split_number)

        print('{}: Finish creating val data features'.format((time.asctime(time.localtime(time.time())))))
        logging.info('Finish creating val data features')

    print('{}: Done!'.format((time.asctime(time.localtime(time.time())))))
    logging.info('Done!')

    return


@ray.remote
def execute_parallel(data_set, data_type: str):
    """
    This function run the process of creating features in parallel
    :param data_set: directory to read the data from
    :param data_type: which data we use- train/test/val
    :return:
    """
    log_file_name = os.path.join(log_directory, datetime.now().strftime(
                                    f'LogFile_create_features_delta_{data_set}_%d_%m_%Y_%H_%M_%S.log'))
    logging.basicConfig(filename=log_file_name,
                        level=logging.DEBUG,
                        format='%(asctime)s: %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        )

    topics_number = 15
    print('{}: Create object'.format((time.asctime(time.localtime(time.time())))))
    logging.info('{}: Create object'.format((time.asctime(time.localtime(time.time())))))
    inner_max_branch_length = max_branch_length_dict[data_type]
    create_features = CreateFeatures(topics_number, inner_max_branch_length)

    print('{}: Loading train data'.format((time.asctime(time.localtime(time.time())))))
    logging.info('{}: Loading train data'.format((time.asctime(time.localtime(time.time())))))
    # load_data to be False if we already have the trained model and we don't need to load the train data
    trained_models_dir = os.path.join(train_test_data_directory, 'split_data', data_set, 'trained_models')
    create_features.create_data('train', is_train=True, load_data=True, trained_models_dir=trained_models_dir)

    print('{}: Start run data set {}'.format((time.asctime(time.localtime(time.time()))), data_set))
    logging.info('Start run data set {}'.format(data_set))

    features_dir_path = os.path.join(features_directory, data_set)
    if not os.path.exists(features_dir_path):
        os.makedirs(features_dir_path)

    curr_data_directory = os.path.join(train_test_data_directory, 'split_data', data_set)

    print('{}: Loading data'.format((time.asctime(time.localtime(time.time())))))
    logging.info('Loading data')
    create_features.create_data(data_type, is_train=False, data_dir=curr_data_directory)

    print('{}: Finish loading the data. Data sizes: {}. Start create features'.
          format((time.asctime(time.localtime(time.time()))), create_features.data.shape))
    logging.info('Finish loading the data. Data sizes: {}. Start create features'.format(create_features.data.shape))

    create_features.create_all_features(features_dir_path)

    print('{}: Finish creating features'.format((time.asctime(time.localtime(time.time())))))
    logging.info('Finish creating features')

    return data_set + ' is ready'


def parallel_main():
    ray.init()

    if len(sys.argv) > 1:
        data_type = sys.argv[2]
        specific_data_type = True
    else:
        specific_data_type = False
        data_type = ''

    print('{}: Start run in parallel for data type {}'.format((time.asctime(time.localtime(time.time()))), data_type))
    logging.info('{}: Start run in parallel for data type {}'.format((time.asctime(time.localtime(time.time()))),
                                                                     data_type))

    # data_dirs = [data for data in os.listdir(os.path.join(train_test_data_directory, 'split_data'))
    #              if data.startswith('train')]
    if not specific_data_type:
        data_dirs = os.listdir(os.path.join(train_test_data_directory, 'split_data'))
    else:
        data_dirs = [data for data in os.listdir(os.path.join(train_test_data_directory, 'split_data'))
                     if data.startswith(data_type)]
    print(f'Directories are: {data_dirs}')

    all_ready_lng = ray.get([execute_parallel.remote(data_set, data_set[:5]) for data_set in data_dirs])

    print('{}: Done! {}'.format((time.asctime(time.localtime(time.time()))), all_ready_lng))
    logging.info('{}: Done! {}'.format((time.asctime(time.localtime(time.time()))), all_ready_lng))

    return


def manual_parallel_main():
    data_set = sys.argv[2]
    print(f'Running on {data_set}')
    logging.info('Running on {}'.format(data_set))

    log_file_name = os.path.join(log_directory, datetime.now().strftime(
        f'LogFile_create_features_delta_{data_set}_%d_%m_%Y_%H_%M_%S.log'))
    logging.basicConfig(filename=log_file_name,
                        level=logging.DEBUG,
                        format='%(asctime)s: %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        )
    topics_number = 15
    print('{}: Create object'.format((time.asctime(time.localtime(time.time())))))
    logging.info('Create object')
    inner_max_branch_length = max_branch_length_dict[data_set[:5]]
    create_features = CreateFeatures(topics_number, inner_max_branch_length)

    print('{}: Loading train data or fitted models'.format((time.asctime(time.localtime(time.time())))))
    logging.info('Loading train data or fitted models')
    # load_data to be False if we already have the trained model and we don't need to load the train data
    trained_models_dir = os.path.join(train_test_data_directory, 'split_data', data_set, 'trained_models')
    curr_data_directory = os.path.join(train_test_data_directory, 'split_data', data_set)

    create_features.create_data('train', is_train=True, load_data=False, trained_models_dir=trained_models_dir,
                                data_dir=curr_data_directory)

    features_dir_path = os.path.join(features_directory, data_set)
    if not os.path.exists(features_dir_path):
        os.makedirs(features_dir_path)

    print('{}: Loading data'.format((time.asctime(time.localtime(time.time())))))
    logging.info('Loading data')
    create_features.create_data(data_set[:5], is_train=False, data_dir=curr_data_directory)

    print('{}: Finish loading the data. Data sizes: {}. Start create features'.
          format((time.asctime(time.localtime(time.time()))), create_features.data.shape))
    logging.info(
        'Finish loading the data. Data sizes: {}. Start create features'.format(create_features.data.shape))

    create_features.create_all_features(features_dir_path)

    print('{}: Finish creating features'.format((time.asctime(time.localtime(time.time())))))
    logging.info('Finish creating features')

    print('{}: Done!'.format((time.asctime(time.localtime(time.time())))))
    logging.info('Done!')

    return


if __name__ == '__main__':
    """
    sys.argv[1] = main_func
    sys.argv[2] = data_dir / data_type to run in ray
    """
    main_func = sys.argv[1]
    print(f'Start run {main_func}')
    logging.info('Start run {}'.format(main_func))

    if main_func == 'parallel_main':
        parallel_main()
    elif main_func == 'not_parallel_main':
        not_parallel_main()
    elif main_func == 'manual_parallel_main':
        manual_parallel_main()
    else:
        print(f'{main_func} is not main function in this code')
        logging.info('{} is not main function in this code'.format(main_func))

