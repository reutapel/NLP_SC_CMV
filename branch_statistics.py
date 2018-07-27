import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np


base_directory = os.path.abspath(os.curdir)
data_directory = os.path.join(base_directory, 'change my view')
branch_statistics_directory = os.path.join(data_directory, 'branch_statistics')


class CalculateStatistics:
    def __init__(self):
        self.statistics = pd.DataFrame(columns=['mean', 'STD', 'sum', 'count', 'min', 'max'])
        self.branch_numbers_df = pd.read_csv(os.path.join(data_directory, 'branch_numbers_df_small.csv'))
        self.branch_comments_info_df =\
            pd.read_csv(os.path.join(data_directory,
                                     'all_submissions_comments_with_label_all_deltalog_final_with_branches_small.csv'),
                        usecols=['branch_id', 'comment_id', 'comment_real_depth', 'delta', 'submission_id',
                                 'comment_author', 'comment_is_submitter'])
        self.statistics_columns = ['branch_length', 'delta_index_in_branch', 'num_comments_after_delta', 'num_delta']

    def calculate_statistics_hist(self, statistics_column_name, is_split=False):
        """
        This function calculate statistics and create histogram for column in branch_numbers_df
        :param str statistics_column_name: the name of the column we want to calculate
        :param bool is_split: if we want to split the data to delta and no delta branches or not
        :return:
        """
        # get column and calculate statistics:
        if not is_split:
            data = self.branch_numbers_df[statistics_column_name].dropna()
            data = data.loc[data != 0]
            statistics_column = [(data, statistics_column_name + '_all')]
        else:
            statistics_column =\
                [((self.branch_numbers_df.loc[self.branch_numbers_df['num_delta'] == 0][statistics_column_name]).dropna(),
                  statistics_column_name + '_no_delta'),
                 ((self.branch_numbers_df.loc[self.branch_numbers_df['num_delta'] > 0][statistics_column_name]).dropna(),
                  statistics_column_name + '_delta')]
        for statistics_column_item, statistics_column_item_name in statistics_column:
            branch_length_stat = pd.DataFrame({'mean': statistics_column_item.mean(),
                                               'STD': statistics_column_item.std(),
                                               'median': statistics_column_item.median(),
                                               'sum': statistics_column_item.sum(),
                                               'count': statistics_column_item.count(),
                                               'min': statistics_column_item.min(),
                                               'max': statistics_column_item.max()},
                                              index=[statistics_column_item_name])
            self.statistics = self.statistics.append(branch_length_stat)[self.statistics.columns.tolist()]
        # create and save histogram
        hist_title = 'Histogram_' + statistics_column_name + '_is_split_' + str(is_split)
        plot = plt.figure(hist_title)
        number_of_bins = int(np.nanmax(statistics_column[0][0].unique())) + 1
        plt.hist(statistics_column[0][0], bins=np.arange(0, number_of_bins, 1), label=statistics_column[0][1],
                 ls='dashed', lw=3, fc=(0, 0, 1, 0.5))
        if is_split:  # add the no_delta to the same hist
            plt.hist(statistics_column[1][0], bins=np.arange(0, number_of_bins, 1), label=statistics_column[1][1],
                     ls='dotted', lw=3, fc=(1, 0, 0, 0.5))
        plt.legend()
        plt.title(hist_title)
        plt.xlabel(statistics_column_name + '_is_split_' + str(is_split))
        plt.ylabel('number of branches')
        fig_to_save = plot
        fig_to_save.savefig(os.path.join(branch_statistics_directory, hist_title + '.png'), bbox_inches='tight')

    def statistics_per_submission(self):
        """
        This function calculate statistics per submission
        :return:
        """
        functions = {'branch_length': ['count', 'mean', 'std', 'median', 'min', 'max']}
        submission_group_by = self.branch_numbers_df.groupby('submission_id').agg(functions)
        # split data to delta and no delta to get statistics
        only_delta = self.branch_numbers_df.loc[self.branch_numbers_df['num_delta'] > 0]
        delta_function = {'num_delta': ['mean', 'std', 'min', 'max'],
                          'delta_index_in_branch': ['count', 'mean', 'median', 'std', 'min', 'max'],
                          'num_comments_after_delta': ['mean', 'median', 'std', 'min', 'max'],
                          'branch_length': ['count', 'mean', 'median', 'std', 'min', 'max']}
        no_delta = self.branch_numbers_df.loc[self.branch_numbers_df['num_delta'] == 0]
        no_delta_function = {'branch_length': ['count', 'mean', 'median', 'std', 'min', 'max']}

        delta_group_by = only_delta.groupby('submission_id').agg(delta_function)
        add_delta = 'delta'
        delta_group_by.columns.set_levels([add_delta + '_num_delta', 'delta_index_in_branch',
                                           'num_comments_after_delta', add_delta + '_branch_length'],
                                          level=0, inplace=True)
        no_delta_group_by = no_delta.groupby('submission_id').agg(no_delta_function)
        no_delta_group_by.columns.set_levels(['no_delta_branch_length'], level=0, inplace=True)
        submission_group_by = pd.concat([submission_group_by, delta_group_by, no_delta_group_by], axis=1)
        submission_group_by.to_csv(os.path.join(branch_statistics_directory, 'statistics_per_submission.csv'))

    def statistics_per_branch_users(self):
        """
        This function calculate statistics for each branch. Statistics regarding the users
        :return:
        """
        # put 'submitter' in comment_author because there are many without comment_author, and we do it per branch
        # so it will be the same author in the branch
        self.branch_comments_info_df.loc[
            self.branch_comments_info_df['comment_is_submitter'], 'comment_author'] = 'submitter'
        # statistics_branch_user = pd.DataFrame(columns=['num_users_in_branch', 'mean_comments_per_user',
        #                                                'STD_comments_per_user', 'max_comments_per_user',
        #                                                'min_comments_per_user', 'percent_submitter_comments'])
        num_users_branch = self.branch_comments_info_df.groupby(['branch_id']).agg({'comment_author': 'count'})
        num_users_branch.columns = ['num_authors_branch']
        num_comments_users_branch = self.branch_comments_info_df.groupby(
            ['branch_id', 'comment_author']).agg({'comment_id': 'count'})
        statistics_comments_users_branch = num_comments_users_branch.groupby(
            'branch_id').agg({'comment_id': ['mean', 'median', 'std', 'max', 'min']})
        statistics_comments_users_branch.columns = ['mean_comments_per_user', 'STD_comments_per_user',
                                                    'max_comments_per_user', 'mim_comments_per_user']
        num_comments_users_branch = self.branch_comments_info_df.groupby(
            ['branch_id', 'comment_is_submitter']).agg({'comment_id': 'count'})
        num_comments_users_branch = num_comments_users_branch.unstack()
        num_comments_users_branch.columns = [False, True]
        num_comments_users_branch = num_comments_users_branch.assign(
            submitter_pcts=(num_comments_users_branch[True]) / num_comments_users_branch.sum(axis=1))
        num_comments_users_branch['submitter_pcts'] = num_comments_users_branch.submitter_pcts.fillna(0)
        submitter_pcts = num_comments_users_branch[['submitter_pcts']]
        statistics_branch_user = pd.concat([num_users_branch, statistics_comments_users_branch, submitter_pcts], axis=1)
        statistics_branch_user.to_csv(os.path.join(branch_statistics_directory, 'statistics_branch_user.csv'))


def main():
    calculate_statistics_obj = CalculateStatistics()
    # for each column calculate statistics and create histogram with and without split to delta and no delta
    for statistic_column in calculate_statistics_obj.statistics_columns:
        calculate_statistics_obj.calculate_statistics_hist(statistic_column)
        if statistic_column == 'branch_length':  # only for branch length this is relevant
            calculate_statistics_obj.calculate_statistics_hist(statistic_column, is_split=True)
    calculate_statistics_obj.statistics.to_csv(os.path.join(branch_statistics_directory, 'statistics.csv'))
    calculate_statistics_obj.statistics_per_submission()
    calculate_statistics_obj.statistics_per_branch_users()


if __name__ == '__main__':
    main()
