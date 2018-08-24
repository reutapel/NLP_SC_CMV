# -*- coding: utf-8 -*-
import pandas as pd
import os
import numpy as np

base_directory = os.path.abspath(os.curdir)
data_directory = os.path.join(base_directory, 'data')
branch_statistics_directory = os.path.join(base_directory, 'branch_statistics')


def unicodetoascii(row):
    text = row['comment_body']

    TEXT = (text.replace('\\xe2\\x80\\x99', "'").
            replace('\\xc3\\xa9', 'e').
            replace('\\xe2\\x80\\x90', '-').
            replace('\\xe2\\x80\\x91', '-').
            replace('\\xe2\\x80\\x92', '-').
            replace('\\xe2\\x80\\x93', '-').
            replace('\\xe2\\x80\\x94', '-').
            replace('\\xe2\\x80\\x94', '-').
            replace('\\xe2\\x80\\x98', "'").
            replace('\\xe2\\x80\\x9b', "'").
            replace('\\xe2\\x80\\x9c', '"').
            replace('\\xe2\\x80\\x9c', '"').
            replace('\\xe2\\x80\\x9d', '"').
            replace('\\xe2\\x80\\x9e', '"').
            replace('\\xe2\\x80\\x9f', '"').
            replace('\\xe2\\x80\\xa6', '...').
            replace('\\xe2\\x80\\xb2', "'").
            replace('\\xe2\\x80\\xb3', "'").
            replace('\\xe2\\x80\\xb4', "'").
            replace('\\xe2\\x80\\xb5', "'").
            replace('\\xe2\\x80\\xb6', "'").
            replace('\\xe2\\x80\\xb7', "'").
            replace('\\xe2\\x81\\xba', "+").
            replace('\\xe2\\x81\\xbb', "-").
            replace('\\xe2\\x81\\xbc', "=").
            replace('\\xe2\\x81\\xbd', "(").
            replace('\\xe2\\x81\\xbe', ")")
            )
    return TEXT


class RemoveCommentsFromData:
    def __init__(self):
        # upload data
        self.submissions = pd.read_csv(os.path.join(data_directory, 'all_submissions_final.csv'))
        self.branch_comments_info_df = pd.read_csv(
            os.path.join(data_directory, 'all_submissions_comments_with_label_all_deltalog_final_with_branches.csv'))
        self.branch_numbers_df = pd.read_csv(os.path.join(data_directory, 'branch_numbers_df_fix.csv'))

        # filter out branches of length 1 and more than 29
        self.branch_numbers_df = self.branch_numbers_df.loc[(self.branch_numbers_df['branch_length'] > 1) &
                                                            (self.branch_numbers_df['branch_length'] < 30)]
        # remove this branch because there is no really a delta
        self.branch_numbers_df = self.branch_numbers_df.loc[self.branch_numbers_df.branch_id != 24158]
        # remove this submission because the deltas there seams to be fake, many deltas with the same text
        self.branch_numbers_df = self.branch_numbers_df.loc[self.branch_numbers_df.submission_id != '6tkjmm']

        # filter comments of those branches
        self.branches_to_use = self.branch_numbers_df['branch_id'].unique().tolist()
        self.branch_comments_info_df = self.branch_comments_info_df.loc[
            self.branch_comments_info_df.branch_id.isin(self.branches_to_use)]
        self.new_branches_data_after_remove = self.branch_numbers_df.copy()
        self.new_branches_data_after_remove = self.new_branches_data_after_remove.assign(num_comments_removed=0)
        self.new_branches_data_after_remove = self.new_branches_data_after_remove.assign(pct_comments_removed=0.0)

        # remove b' from comment body
        self.branch_comments_info_df['comment_body'] = self.branch_comments_info_df.comment_body.str.rstrip('"')
        self.branch_comments_info_df['comment_body'] = self.branch_comments_info_df.comment_body.str.replace('b"', '')
        self.branch_comments_info_df['comment_body'] = self.branch_comments_info_df.comment_body.str.rstrip("'")
        self.branch_comments_info_df['comment_body'] = self.branch_comments_info_df.comment_body.str.replace("b'", "")
        self.branch_comments_info_df['comment_body'] = self.branch_comments_info_df.apply(unicodetoascii, axis=1)
        self.new_comments_data_after_remove = self.branch_comments_info_df.copy()

        # define delta tokens
        self.delta_tokens = ['&amp;#8710;', '&#8710;', '&#916;', '&amp;916;', '∆', '!delta', 'Δ', '&delta;',
                             '\\xce\\x94']
        self.after_delta_tokens = ['delta', 'thank', 'thanks']
        self.all_comments_after_giving_delta_author = pd.DataFrame()

        return

    def remove_comments_from_data(self):
        """
        This function check for each comment if we should remove it: it gave a delta or it was removed.
        Update comment_real_depth for comments after the removed comment in the branch.
        Update branch info
        :return:
        """
        branches_removed = list()
        for branch in self.branches_to_use:
            print('branch:', branch)
            if branch in branches_removed:
                continue
            branch_update = False
            comments_in_branch = self.branch_comments_info_df.loc[self.branch_comments_info_df.branch_id == branch]
            for index, comment in comments_in_branch.iterrows():
                # check if the comment gave delta or was removed
                if any(substring in comment['comment_body'] for substring in ['[deleted]', '[removed]']):
                    if comment.delta == 1:
                        # print('Comment', comment.comment_id, 'was removed with delta.'
                        #                                      'Comment body is:', comment.comment_body)
                        # remove branch if the delta is deleted:
                        branches_delta_deleted = self.branch_comments_info_df.loc[
                            self.branch_comments_info_df.comment_id == comment.comment_id]['branch_id'].unique()
                        self.new_comments_data_after_remove = self.new_comments_data_after_remove.loc[
                            ~self.new_comments_data_after_remove.branch_id.isin(branches_delta_deleted)]
                        self.new_branches_data_after_remove = self.new_branches_data_after_remove.loc[
                            ~self.new_branches_data_after_remove.branch_id.isin(branches_delta_deleted)]
                        branches_removed += list(branches_delta_deleted)
                        branch_update = False
                        break
                    # update that the branch was updated:
                    branch_update = True
                    # 1. remove comment from data
                    self.new_comments_data_after_remove = self.new_comments_data_after_remove.drop([index])
                    comments_in_branch = comments_in_branch.drop([index])
                    # 2. update comment_real_depth for comments after the removed comment in the branch
                    comments_to_update = comments_in_branch.loc[comments_in_branch.comment_real_depth >
                                                                comment.comment_real_depth]
                    for update_comment_index, update_comment in comments_to_update.iterrows():
                        self.new_comments_data_after_remove.loc[update_comment_index, 'comment_real_depth'] -= 1

                # check if this comment gave a delta
                elif comment.comment_is_submitter and len(comment['comment_body']) > 50 and\
                        any(delta_token in str(comment['comment_body']) for delta_token in self.delta_tokens):
                    # filter the delta from the comment_body:
                    split_comment = str(comment['comment_body']).split(sep='\\n')
                    # keep only parts without delta token
                    split_comment = [part for part in split_comment if not
                                     any(delta_token in part for delta_token in self.delta_tokens)]
                    split_comment = [part for part in split_comment if part not in ['', '___']]
                    new_comment = '\n'.join(split_comment)
                    # change the comment_body in the DF:
                    if len(split_comment) > 0:  # didn't remove all comment
                        self.new_comments_data_after_remove.loc[index, 'comment_body'] = new_comment
                        self.new_comments_data_after_remove.loc[index, 'before_filter_comment_body'] = \
                            str(comment['comment_body'])
                    else:
                        # 1. remove comment from data
                        # give chance to short comments
                        split_comment = str(comment['comment_body']).split(sep='.')
                        split_comment = [part for part in split_comment if not any(
                            delta_token in part for delta_token in self.delta_tokens)]
                        split_comment = [part for part in split_comment if part not in ['', '___']]
                        new_comment = '.'.join(split_comment)
                        if len(split_comment) > 0:  # didn't remove all comment
                            self.new_comments_data_after_remove.loc[index, 'comment_body'] = new_comment
                            self.new_comments_data_after_remove.loc[index, 'before_filter_comment_body'] = \
                                str(comment['comment_body'])
                        else:
                            # print('comment gave delta removed with comment body:', comment.comment_body)
                            self.new_comments_data_after_remove = \
                                self.new_comments_data_after_remove.drop([index])
                            comments_in_branch = comments_in_branch.drop([index])
                            # 2. update comment_real_depth for comments after the removed comment in the branch
                            comments_to_update = comments_in_branch.loc[
                                comments_in_branch.comment_real_depth > comment.comment_real_depth]
                            for update_comment_index, update_comment in comments_to_update.iterrows():
                                self.new_comments_data_after_remove.loc[
                                    update_comment_index, 'comment_real_depth'] -= 1

                    # check if the next comment is thanks comment by the user got delta:
                    branch_length =\
                        int(self.branch_numbers_df.loc[self.branch_numbers_df.branch_id == branch]['branch_length'])
                    num_delta_in_branch = int(self.branch_numbers_df.loc[
                        self.branch_numbers_df.branch_id == branch]['num_delta'])
                    if (comment.comment_real_depth > 0) and (comment.comment_real_depth < branch_length - 1) and\
                            (num_delta_in_branch > 0):
                        # this is not the first comment in the branch and not the last one
                        # and there deltas in this branch
                        comment_got_delta = comments_in_branch.loc[comments_in_branch.comment_real_depth ==
                                                                   comment.comment_real_depth - 1]
                        comment_after_giving_delta = comments_in_branch.loc[comments_in_branch.comment_real_depth ==
                                                                            comment.comment_real_depth + 1]
                        if int(comment_got_delta.delta) == 1:  # the comment before the current comment got delta
                            comment_got_delta_author = comment_got_delta.comment_author
                            comment_after_giving_delta_author = comment_after_giving_delta.comment_author
                            # the same user got delta and responded to the delta and wrote one of the after_delta_tokens
                            if (comment_got_delta_author.values == comment_after_giving_delta_author.values) and\
                                    (any(delta_token in comment_after_giving_delta['comment_body'][
                                        comment_after_giving_delta.index[0]].lower()
                                         for delta_token in self.after_delta_tokens)):
                                # remove after_delta_tokens from comment
                                # filter the delta from the comment_body:
                                split_comment = comment_after_giving_delta['comment_body'].values[0].split(sep='\n')
                                # keep only parts without delta token
                                split_comment = [part for part in split_comment if
                                                 not any(delta_token in part for delta_token in self.after_delta_tokens)]
                                new_comment = '\n'.join(split_comment)
                                comment_after_giving_delta = comment_after_giving_delta.assign(new_comment=new_comment)
                                self.all_comments_after_giving_delta_author = \
                                    self.all_comments_after_giving_delta_author.append(comment_after_giving_delta)
                                self.all_comments_after_giving_delta_author.to_csv(os.path.join(
                                    data_directory, 'all_comments_after_giving_delta_author.csv'))
                                # change the comment_body in the DF:
                                if len(split_comment) > 0:  # didn't remove all comment
                                    self.new_comments_data_after_remove.loc[comment_got_delta.index, 'comment_body'] =\
                                        new_comment
                                    # print(comment_got_delta.comment_body.values[0])
                                    self.new_comments_data_after_remove.loc[comment_got_delta.index,
                                                                            'before_filter_comment_body'] =\
                                        comment_got_delta.comment_body.values[0]
                                else:
                                    # 1. remove comment from data
                                    # give chance to short comments
                                    split_comment = str(comment_after_giving_delta['comment_body']).split(sep='.')
                                    split_comment = [part for part in split_comment if not any(
                                        delta_token in part for delta_token in self.delta_tokens)]
                                    split_comment = [part for part in split_comment if part not in ['', '___']]
                                    new_comment = '.'.join(split_comment)
                                    if len(split_comment) > 0:  # didn't remove all comment
                                        self.new_comments_data_after_remove.loc[
                                            comment_got_delta.index, 'comment_body'] = new_comment
                                        # print(comment_got_delta.comment_body.values[0])
                                        self.new_comments_data_after_remove.loc[comment_got_delta.index,
                                                                                'before_filter_comment_body'] = \
                                            comment_got_delta.comment_body.values[0]
                                    else:
                                        # print('comment after delta removed with comment body:',
                                        #       comment_after_giving_delta['comment_body'].values[0])
                                        self.new_comments_data_after_remove = \
                                            self.new_comments_data_after_remove.drop(comment_after_giving_delta.index)
                                        comments_in_branch = comments_in_branch.drop(comment_after_giving_delta.index)
                                        # 2. update real_depth for comments after the removed comment in the branch
                                        comments_to_update = comments_in_branch.loc[
                                            comments_in_branch.comment_real_depth >
                                            comment_after_giving_delta.comment_real_depth.values[0]]
                                        for update_comment_index, update_comment in comments_to_update.iterrows():
                                            self.new_comments_data_after_remove.loc[
                                                update_comment_index, 'comment_real_depth'] -= 1

            # update branch info
            if branch_update:
                # get the new branch, after removing comments
                update_comments_in_branch =\
                    self.new_comments_data_after_remove.loc[self.new_comments_data_after_remove.branch_id == branch]

                # update branch length:
                branch_length = self.branch_numbers_df.loc[
                    self.branch_numbers_df['branch_id'] == branch]['branch_length']
                self.new_branches_data_after_remove.loc[self.new_branches_data_after_remove.branch_id == branch,
                                                        'branch_length'] = update_comments_in_branch.shape[0]
                num_comments_removed = branch_length - update_comments_in_branch.shape[0]
                self.new_branches_data_after_remove.loc[self.new_branches_data_after_remove.branch_id == branch,
                                                        'num_comments_removed'] = num_comments_removed
                self.new_branches_data_after_remove.loc[self.new_branches_data_after_remove.branch_id == branch,
                                                        'pct_comments_removed'] = num_comments_removed/branch_length

                # update num_comments_after_delta and delta_index_in_branch
                delta_comment = update_comments_in_branch.loc[update_comments_in_branch['delta'] == 1]
                if delta_comment.shape[0] == 1:  # only 1 delta in branch
                    delta_row_num = delta_comment.comment_real_depth + 1  # add 1 because real_depth starts from 0
                    num_comments_after_delta = int(update_comments_in_branch.shape[0] - delta_row_num)
                elif delta_comment.shape[0] > 1:  # more than 1 delta in the branch
                    # get all the indexes of the delta in the branch
                    delta_row_list = list(delta_comment.comment_real_depth)
                    # get the last index
                    delta_row_num = delta_row_list[-1] + 1  # add 1 because the index starts from 0
                    num_comments_after_delta = int(update_comments_in_branch.shape[0] - delta_row_num)
                # if there are no deltas- no need to update this
                else:
                    continue
                # update branches info:
                self.new_branches_data_after_remove.loc[self.new_branches_data_after_remove.branch_id == branch,
                                                        'num_comments_after_delta'] = num_comments_after_delta
                self.new_branches_data_after_remove.loc[self.new_branches_data_after_remove.branch_id == branch,
                                                        'delta_index_in_branch'] = delta_row_num

        return

    def create_final_data(self):
        """
        This function join 3 data frames to one that will be used for the features creation
        :return:
        """
        self.new_comments_data_after_remove.to_csv(os.path.join(data_directory, 'new_comments_data_after_remove.csv'))
        branch_relevant_info = self.new_branches_data_after_remove['branch_id', 'branch_length', 'num_delta',
                                                                   'num_comments_after_delta', 'delta_index_in_branch']
        join_data = self.new_comments_data_after_remove.merge(branch_relevant_info, on='branch_id')
        join_data = join_data.merge(self.submissions, on='submission_id')

        join_data = join_data.sort_values(by=['branch_id', 'comment_real_depth'], ascending=[False, True])

        join_data.to_csv(os.path.join(data_directory, 'comments_label_branch_info_after_remove.csv'))


def main():
    remove_comments_obj = RemoveCommentsFromData()
    remove_comments_obj.remove_comments_from_data()
    remove_comments_obj.create_final_data()


if __name__ == '__main__':
    main()
