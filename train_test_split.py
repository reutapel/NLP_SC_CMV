import pandas as pd
from time import time
import time
import math
from datetime import datetime
import logging
import pytz
from copy import copy
import os
import nltk as nk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
# import gensim
from gensim import corpora
# from nltk.stem import PorterStemmer
# from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.sklearn_api import ldamodel
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer


base_directory = os.path.abspath(os.curdir)
data_directory = os.path.join(base_directory, 'data')

log_directory = os.path.join(base_directory, 'logs')
LOG_FILENAME = os.path.join(log_directory,
                            datetime.now().strftime('LogFile_create_features_delta_%d_%m_%Y_%H_%M_%S.log'))
logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO, )


def branch_group_size(row):
    if row['branch_length'] <= 12:
        return 'small'
    elif row['branch_length'] <= 21:
        return 'medium'
    else:
        return 'large'


def assign_submission_id(row):
    return row.name[0]


def small_delta(row):
    if row['branch_id'] <= 4:
        return 1
    elif row['branch_id'] <= 20:
        return 2
    else:
        return 3


def small_no_delta(row):
    if row['branch_id'] <= 3:
        return 4
    elif row['branch_id'] <= 8:
        return 5
    else:
        return 6


def medium_delta(row):
    if row['branch_id'] <= 2:
        return 7
    else:
        return 8


def medium_no_delta(row):
    if row['branch_id'] <= 3:
        return 9
    elif row['branch_id'] <= 8:
        return 10
    else:
        return 11


def large_delta(row):
    if row['branch_id'] <= 1:
        return 12
    else:
        return 13


def large_no_delta(row):
    if row['branch_id'] <= 1:
        return 14
    else:
        return 15


group_dict = {
    'small_delta': small_delta,
    'small_no_delta': small_no_delta,
    'medium_delta': medium_delta,
    'medium_no_delta': medium_no_delta,
    'large_delta': large_delta,
    'large_no_delta': large_no_delta,
}


def train_test_split():
    branch_comments_info_df = pd.read_csv(
        os.path.join(data_directory, 'all_submissions_comments_with_label_all_deltalog_final_with_branches.csv'))
    submissions = pd.read_csv(os.path.join(data_directory, 'all_submissions_final.csv'))
    branch_comments_info_df = branch_comments_info_df.merge(submissions, on='submission_id')

    branch_numbers_df = pd.read_csv(os.path.join(data_directory, 'branch_numbers_df_fix.csv'))
    # filter out branches of length 1 and more than 29
    branch_numbers_df = branch_numbers_df.loc[(branch_numbers_df['branch_length'] > 1) &
                                              (branch_numbers_df['branch_length'] < 30)]
    # remove this branch because there is no really a delta
    branch_numbers_df = branch_numbers_df.loc[branch_numbers_df.branch_id != 24158]
    # remove this submission because the deltas there seams to be fake, many deltas with the same text
    branch_numbers_df = branch_numbers_df.loc[branch_numbers_df.submission_id != '6tkjmm']
    # assign branch_length_group:
    branch_numbers_df['branch_length_group'] = branch_numbers_df.apply(branch_group_size, axis=1)

    # functions = {'branch_length': {'num_branches': 'count'},
    #              'num_delta': {'num_deltas': 'sum'}}
    # sub_group_by = branch_numbers_df.groupby('submission_id').agg(functions)
    submission_group = pd.DataFrame(columns=['group_number', 'submission_id'])
    sub_branch_size_group_by = branch_numbers_df.groupby(['submission_id', 'branch_length_group', 'num_delta']).\
        agg({'branch_id': 'count'})

    sub_branch_size_group_by.to_csv(os.path.join(data_directory, 'sub_branch_size_group_by.csv'))

    for group in ['small', 'medium', 'large']:
        # assign group number for no delta
        no_delta = sub_branch_size_group_by.iloc[
            (sub_branch_size_group_by.index.get_level_values('branch_length_group') == group) &
            (sub_branch_size_group_by.index.get_level_values('num_delta') == 0)]
        to_append = pd.DataFrame(no_delta.apply(group_dict[group + '_no_delta'], axis=1), columns=['group_number'])
        to_append['submission_id'] = to_append.apply(assign_submission_id, axis=1)
        submission_group = submission_group.append(to_append)

        # assign group number for delta
        delta = sub_branch_size_group_by.iloc[
            (sub_branch_size_group_by.index.get_level_values('branch_length_group') == group) &
            (sub_branch_size_group_by.index.get_level_values('num_delta') > 0)]
        to_append = pd.DataFrame(delta.apply(group_dict[group + '_delta'], axis=1), columns=['group_number'])
        to_append['submission_id'] = to_append.apply(assign_submission_id, axis=1)
        submission_group = submission_group.append(to_append)

    # get the submissions for train data
    all_submissions = list(submission_group.submission_id.unique())
    train_submissions = list()
    for group_number in range(1, 16):
        rows_in_group = submission_group.loc[submission_group.group_number == group_number]
        num_to_sample = math.floor(0.4 * rows_in_group.shape[0])
        train_sample = rows_in_group.sample(n=num_to_sample)
        train_submissions += list(train_sample.submission_id.unique())

    # get the users in the train submissions
    train_data = branch_comments_info_df.loc[branch_comments_info_df.submission_id.isin(train_submissions)]
    train_data_users = list(train_data['comment_author'])
    train_data_users.append(list(train_data['submission_author']))
    train_data_users = [user for user in train_data_users if str(user) != 'nan']
    branch_comments_info_df.comment_author = branch_comments_info_df.comment_author.to_string()
    final_train_data = branch_comments_info_df.loc[
        branch_comments_info_df.comment_author.isin(train_data_users)]
    final_train_data = final_train_data.append(branch_comments_info_df.loc[
        branch_comments_info_df.submission_author.isin(train_data_users)])
    final_train_data.to_csv(os.path.join(data_directory, 'train_data.csv'))

    # create test and validation data
    submissions_final_train_data = list(final_train_data['submission_id'])
    submissions_not_in_train = np.setdiff1d(all_submissions, submissions_final_train_data)
    num_to_sample = int(math.floor(0.5 * submissions_not_in_train.shape[0]))
    test_submissions = np.random.choice(submissions_not_in_train, size=num_to_sample)
    test_data = branch_comments_info_df.loc[branch_comments_info_df.submission_id.isin(test_submissions)]
    test_data.to_csv(os.path.join(data_directory, 'test_data.csv'))
    val_submissions = np.setdiff1d(submissions_not_in_train, test_submissions)
    val_data = branch_comments_info_df.loc[branch_comments_info_df.submission_id.isin(val_submissions)]
    val_data.to_csv(os.path.join(data_directory, 'val_data.csv'))


def main():
    train_test_split()


if __name__ == '__main__':
    main()