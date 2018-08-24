import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np


base_directory = os.path.abspath(os.curdir)
data_directory = os.path.join(base_directory, 'data')
branch_statistics_directory = os.path.join(base_directory, 'branch_statistics')


def autolabel(rects, ax, rotation):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 5, height, ha='center', va='bottom',
                rotation=rotation)


def create_chart_bars(title, x, y, xlabel, ylabel, y2=None, x2=None, is_split=False):
    """
    This function create a chart bar with number of samples in each bin
    :param str title: title of the bar
    :param numpy array x: the values in the x axis
    :param list y: the values in the y axis
    :param str xlabel: the label of the x axis
    :param str ylabel: the label of the y axis
    :param list y2: the values in the y axis if we want to plot 2 lists on the same bar
    :param numpy array x2: the values in the x axis if we want to plot 2 lists on the same bar
    :param bool is_split: if we have 2 lists on the same bar
    :return:
    """

    fig, ax = plt.subplots()
    width = 0.35
    print('Create bar for', title)

    if np.max(x) < 10:
        rotation = 'horizontal'
    else:
        rotation = 'vertical'

    if is_split:  # add the no_delta to the same hist
        rects1 = ax.bar(x, y, width, color='b')
        rects2 = ax.bar(x2 + width, y2, width, color='y')
        ax.legend((rects1[0], rects2[0]), ('no_delta', 'delta'))
        autolabel(rects1, ax, rotation)
        autolabel(rects2, ax, rotation)
    else:
        plt.bar(x, y)
        rects = ax.patches
        autolabel(rects, ax, rotation)

    # add some text for labels, title and axes ticks
    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(x)

    plt.show()

    fig_to_save = fig
    fig_to_save.savefig(os.path.join(branch_statistics_directory, title + '.png'), bbox_inches='tight')

    return


class CalculateStatistics:
    def __init__(self, drop_1_length=True):
        """
        initiate the class
        :param bool drop_1_length: if to include branches with length 1
        :return:
        """
        self.statistics = pd.DataFrame(columns=['mean', 'STD', 'median', 'sum', 'count', 'min', 'max'])
        self.branch_numbers_df = pd.read_csv(os.path.join(data_directory, 'branch_numbers_df_fix.csv'))

        if drop_1_length:
            self.branch_numbers_df = self.branch_numbers_df.loc[(self.branch_numbers_df['branch_length'] > 1) &
                                                                (self.branch_numbers_df['branch_length'] < 30)]
        self.branch_comments_info_df =\
            pd.read_csv(os.path.join(data_directory,
                                     'all_submissions_comments_with_label_all_deltalog_final_with_branches.csv'),
                        usecols=['branch_id', 'comment_id', 'comment_real_depth', 'delta', 'submission_id',
                                 'comment_author', 'comment_is_submitter', 'comment_body'])
        print('shape with all comments', self.branch_comments_info_df.shape)
        number_of_comments = self.branch_comments_info_df.comment_id.unique()
        print('number of comments before filter', number_of_comments.shape)
        if drop_1_length:
            branches_to_use = self.branch_numbers_df['branch_id'].unique().tolist()
            self.branch_comments_info_df = self.branch_comments_info_df.loc[
                self.branch_comments_info_df.branch_id.isin(branches_to_use)]
            print('shape after filter', self.branch_comments_info_df.shape)
            number_of_comments = self.branch_comments_info_df.comment_id.unique()
            print('number of comments after filter', number_of_comments.shape)

        self.statistics_columns = ['branch_length', 'delta_index_in_branch', 'num_comments_after_delta', 'num_delta']

    def calculate_statistics_hist(self, statistics_column_name, is_split=False, per_branch_length=False,
                                  branch_length=0):
        """
        This function calculate statistics and create histogram for column in branch_numbers_df
        :param str statistics_column_name: the name of the column we want to calculate
        :param bool is_split: if we want to split the data to delta and no delta branches or not
        :param bool per_branch_length: if we calculate the statistics per branch length
        :param int branch_length: the branch length if per_branch_length = True
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
        if per_branch_length:
            data = self.branch_numbers_df.loc[self.branch_numbers_df['branch_length'] == branch_length]
            data = data[statistics_column_name].dropna()
            data = data.loc[data != 0]
            if data.empty:
                print('No delta for branch length', branch_length)
                return
            statistics_column = [(data, statistics_column_name + '_for_branch_length_' + str(branch_length))]

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
        title = 'Bar_' + statistics_column_name + '_is_split_' + str(is_split)
        if per_branch_length:
            title = 'Bar_' + statistics_column_name + '_for_branch_length_' + str(branch_length)

        number_of_bins = int(np.nanmax(statistics_column[0][0].unique())) + 1
        # start_bins = int(np.nanmin(statistics_column[0][0].unique()))
        if statistics_column_name in ['branch_length', 'num_comments_after_delta', 'delta_index_in_branch']\
                and number_of_bins > 20 and not per_branch_length:
            number_of_bins = 20

        data_to_plot = pd.DataFrame(statistics_column[0][0], columns=[statistics_column_name]).assign(count=1)
        data_to_plot = data_to_plot.loc[data_to_plot[statistics_column_name] <= number_of_bins]
        data_to_plot = data_to_plot.groupby(statistics_column_name).count()
        x = data_to_plot.index.values
        y = data_to_plot.iloc[0:number_of_bins]['count'].tolist()

        if is_split:  # add the no_delta to the same hist
            data_to_plot = pd.DataFrame(statistics_column[1][0], columns=[statistics_column_name]).assign(count=1)
            data_to_plot = data_to_plot.loc[data_to_plot[statistics_column_name] <= number_of_bins]
            data_to_plot = data_to_plot.groupby(statistics_column_name).count()
            y2 = data_to_plot.iloc[0:number_of_bins]['count'].tolist()
            x2 = data_to_plot.index.values
            create_chart_bars(title, x, y, xlabel=statistics_column_name + '_is_split_' + str(is_split),
                              ylabel='number of branches', y2=y2, x2=x2, is_split=is_split)

        else:
            create_chart_bars(title, x, y, xlabel=statistics_column_name + '_is_split_' + str(is_split),
                              ylabel='number of branches')

        return

    def statistics_per_submission(self):
        """
        This function calculate statistics per submission
        :return:
        """
        functions = {'branch_length': ['count', 'mean', 'std', 'median', 'min', 'max']}
        submission_group_by = self.branch_numbers_df.groupby('submission_id').agg(functions)
        # split data to delta and no delta to get statistics
        only_delta = self.branch_numbers_df.loc[self.branch_numbers_df['num_delta'] > 0]
        delta_function = {'num_delta': ['count', 'mean', 'median', 'std', 'min', 'max'],
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
        statistics_comments_users_branch.columns = ['mean_comments_per_user', 'median_comments_per_user',
                                                    'STD_comments_per_user', 'max_comments_per_user',
                                                    'mim_comments_per_user']
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

    def index_delta_branch_length(self):
        """
        This function calculate the index of the delta per branch length
        :return:
        """
        branch_length = self.branch_numbers_df['branch_length'].unique().tolist()
        for length in branch_length:
            self.calculate_statistics_hist('delta_index_in_branch', per_branch_length=True, branch_length=length)

        return

    def index_delta_per_comment(self):
        only_deltas = self.branch_comments_info_df.loc[self.branch_comments_info_df['delta'] == 1]
        only_deltas.drop_duplicates(subset=['comment_id'], inplace=True)
        title = 'delta_index_per_comment'
        statistics_column_name = 'comment_real_depth'

        number_of_bins = int(np.nanmax(only_deltas[statistics_column_name].unique())) + 1

        number_of_comments = only_deltas.comment_id.unique()
        print('number of comments with delta', number_of_comments.shape)

        data_to_plot = only_deltas[[statistics_column_name]].assign(count=1)
        data_to_plot = data_to_plot.loc[data_to_plot[statistics_column_name] <= number_of_bins]
        data_to_plot = data_to_plot.groupby(statistics_column_name).count()
        x = data_to_plot.index.values
        y = data_to_plot.iloc[0:number_of_bins]['count'].tolist()

        create_chart_bars('Bar_' + title, x, y, xlabel=title, ylabel='number of comments')


def main():
    calculate_statistics_obj = CalculateStatistics()
    calculate_statistics_obj.index_delta_per_comment()
    calculate_statistics_obj.index_delta_branch_length()

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
