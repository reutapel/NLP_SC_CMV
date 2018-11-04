import pandas as pd
import math
from datetime import datetime
import logging
import os
import numpy as np


base_directory = os.path.abspath(os.curdir)
data_directory = os.path.join(base_directory, 'data')
save_data_directory = os.path.join(data_directory, 'filter_submissions')

log_directory = os.path.join(base_directory, 'logs')
LOG_FILENAME = os.path.join(log_directory,
                            datetime.now().strftime('LogFile_create_features_delta_%d_%m_%Y_%H_%M_%S.log'))
logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO, )

delta_groups = [1, 2, 3, 7, 8, 12, 13]


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
    elif row['branch_id'] <= 21:
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
    # load comments and branches
    branch_comments_info_df = pd.read_csv(
        os.path.join(save_data_directory, 'comments_label_branch_info_after_remove.csv'))
    branch_numbers_df = pd.read_csv(os.path.join(save_data_directory, 'new_branches_data_after_remove.csv'))
    branch_numbers_df = branch_numbers_df.loc[(branch_numbers_df['branch_length'] > 1)]
    branches_to_use = branch_numbers_df.branch_id.unique()
    branch_comments_info_df = branch_comments_info_df.loc[branch_comments_info_df.branch_id.isin(branches_to_use)]

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
        print('group size:', rows_in_group.shape)
        if group_number in delta_groups:
            num_to_sample = math.floor(0.2 * rows_in_group.shape[0])
        else:
            num_to_sample = math.floor(0.32 * rows_in_group.shape[0])
        print('num_to_sample:', num_to_sample)
        train_sample = rows_in_group.sample(n=num_to_sample)
        train_submissions += list(train_sample.submission_id.unique())

    # get the users in the train submissions
    train_data = branch_comments_info_df.loc[branch_comments_info_df.submission_id.isin(train_submissions)]
    # train_data_users = list(train_data['comment_author'])
    # train_data_users.append(list(train_data['submission_author']))
    # train_data_users = [user for user in train_data_users if str(user) != 'nan']
    # branch_comments_info_df.comment_author = branch_comments_info_df.comment_author.to_string()
    # final_train_data = branch_comments_info_df.loc[
    #     branch_comments_info_df.comment_author.isin(train_data_users)]
    # final_train_data = final_train_data.append(branch_comments_info_df.loc[
    #     branch_comments_info_df.submission_author.isin(train_data_users)])
    print('save train data')
    train_branches = train_data.branch_id.unique()
    print('number of branches', train_branches.shape[0])
    num_deltas_train = train_data.loc[train_data.num_delta > 0].branch_id.unique()
    print('number of branches with delta', num_deltas_train.shape[0])
    train_data.to_csv(os.path.join(save_data_directory, 'train_data.csv'))

    # create test and validation data
    # get the submissions for test data
    submissions_final_train_data = list(train_data['submission_id'])
    submissions_not_in_train = np.setdiff1d(all_submissions, submissions_final_train_data)
    submission_group_not_in_train = submission_group.loc[submission_group.submission_id.isin(submissions_not_in_train)]
    test_submissions = list()
    for group_number in range(1, 16):
        rows_in_group = submission_group_not_in_train.loc[submission_group_not_in_train.group_number == group_number]
        print('group size:', rows_in_group.shape)
        if group_number in delta_groups:
            num_to_sample = math.floor(0.2 * rows_in_group.shape[0])
        else:
            num_to_sample = math.floor(0.35 * rows_in_group.shape[0])
        print('num_to_sample:', num_to_sample)
        test_sample = rows_in_group.sample(n=num_to_sample)
        test_submissions += list(test_sample.submission_id.unique())

    # get the users in the test submissions
    test_data = branch_comments_info_df.loc[branch_comments_info_df.submission_id.isin(test_submissions)]
    print('save test data')
    test_branches = test_data.branch_id.unique()
    print('number of branches', test_branches.shape[0])
    num_deltas_test = test_data.loc[test_data.num_delta > 0].branch_id.unique()
    print('number of branches with delta', num_deltas_test.shape[0])
    test_data.to_csv(os.path.join(save_data_directory, 'test_data.csv'))

    val_submissions = np.setdiff1d(submissions_not_in_train, test_submissions)
    val_data = branch_comments_info_df.loc[branch_comments_info_df.submission_id.isin(val_submissions)]
    print('save val data')
    val_branches = val_data.branch_id.unique()
    print('number of branches', val_branches.shape[0])
    num_deltas_val = val_data.loc[val_data.num_delta > 0].branch_id.unique()
    print('number of branches with delta', num_deltas_val.shape[0])
    val_data.to_csv(os.path.join(save_data_directory, 'val_data.csv'))


def main():
    train_test_split()


if __name__ == '__main__':
    main()
