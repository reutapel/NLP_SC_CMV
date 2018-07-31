import pandas as pd
from collections import defaultdict
from copy import copy
import pickle
import numpy as np
import os
import time

base_directory = os.path.abspath(os.curdir)
data_directory = os.path.join(base_directory, 'data')
original_data_file_path = os.path.join(data_directory, 'all_submissions_comments_with_label_all_deltalog_final.csv')


class BranchStatistics:
    def __init__(self):
        """
        Create the class parameters
        """
        # read data
        print('{}: Read data and create class parameters'.format((time.asctime(time.localtime(time.time())))))
        self.comments_with_label = pd.read_csv(original_data_file_path)

        # format of self.branch_info_dict and df:
        # {submission_id: {comment_id of the branch concat with _: [[list of all comments in branch (including root):
        #                                          (comment_id, comment_depth, delta)],
        #                                          branch_length, num_delta, num_comments_after_delta,
        #                                          delta_index_in_branch]
        self.branch_info_dict = defaultdict(dict)
        # df with the numbers of each branch for statistics analysis
        self.branch_numbers_df = pd.DataFrame(columns=['submission_id', 'branch_id', 'branch_key', 'branch_length',
                                                       'num_delta', 'num_comments_after_delta',
                                                       'delta_index_in_branch'])
        # df with row for each comment in the data with its branch info
        self.branch_comments_info_df = pd.DataFrame(columns=['submission_id', 'branch_id', 'comment_id',
                                                             'comment_real_depth', 'delta'])
        # hash table for branch_id and the branch_key
        self.branch_key_branch_id_hash_table = dict()
        self.branch_id_branch_key_hash_table = dict()

        # format of self.root_info_dict:
        # {submission_id: {root_id: [number of branches in root, number of delta in root, number of branches with delta,
        #                            number of branches with comments after delta]
        self.root_info_dict = defaultdict(dict)
        self.root_info_df = pd.DataFrame(columns=['submission_id', 'root_id', 'num_branches_in_root',
                                                  'num_deltas_in_root', 'num_branches_with_delta_in_root',
                                                  'num_branches_comments_after_delta_in_root'])

        self.num_branches_in_root = 0
        self.num_deltas_in_root = 0
        self.num_branches_with_delta_in_root = 0
        self.num_branches_comments_after_delta_in_root = 0
        self.number_branches = 0
        self.number_roots = 0
        self.number_submissions = 0

        # clean comment_ids from b'
        self.comments_with_label['comment_id'] = self.comments_with_label.comment_id.str.rstrip("'")
        self.comments_with_label['comment_id'] = self.comments_with_label.comment_id.str.replace("b'", "")
        self.comments_with_label['submission_id'] = self.comments_with_label.submission_id.str.rstrip("'")
        self.comments_with_label['submission_id'] = self.comments_with_label.submission_id.str.replace("b'", "")
        self.comments_with_label['parent_id'] = self.comments_with_label.parent_id.str.replace("b't3_", "")
        self.comments_with_label['parent_id'] = self.comments_with_label.parent_id.str.replace("b't1_", "")
        self.comments_with_label['parent_id'] = self.comments_with_label.parent_id.str.replace("'", "")

        # remove delta bot from the data:
        self.comments_with_label = self.comments_with_label.loc[
            self.comments_with_label['comment_author'] != 'DeltaBot']

        # DF with all the comments that are not roots
        self.not_roots = copy(self.comments_with_label.loc[~self.comments_with_label['comment_is_root']])
        self.not_roots = self.not_roots[['comment_id', 'submission_id', 'comment_depth', 'delta', 'parent_id']]
        self.submission_comments_df = None

    def insert_row_root_df(self, submission_id, root_id):
        """
        This function insert new row of info to a data frame
        :param: str submission_id: the submission id of the row
        :param: str root_id: the root id of the row
        :return:
        """
        self.root_info_dict[submission_id][root_id] = [self.num_branches_in_root, self.num_deltas_in_root,
                                                       self.num_branches_with_delta_in_root,
                                                       self.num_branches_comments_after_delta_in_root]
        root_row_info = pd.DataFrame({'submission_id': submission_id, 'root_id': root_id,
                                      'num_branches_in_root': self.num_branches_in_root,
                                      'num_deltas_in_root': self.num_deltas_in_root,
                                      'num_branches_with_delta_in_root':
                                          self.num_branches_with_delta_in_root,
                                      'num_branches_comments_after_delta_in_root':
                                          self.num_branches_comments_after_delta_in_root},
                                     index=[self.number_roots])
        self.number_roots += 1
        if self.number_roots % 100 == 0:
            print('{}: Done {} roots'.format((time.asctime(time.localtime(time.time()))), self.number_roots))

        self.root_info_df = self.root_info_df.append(root_row_info)

        # initialize the "root parameters" before we start the next root:
        self.num_branches_in_root = 0
        self.num_deltas_in_root = 0
        self.num_branches_with_delta_in_root = 0
        self.num_branches_comments_after_delta_in_root = 0

        return

    def insert_row_branch_df(self, branch_key, submission_id, branch_df, num_delta, num_comments_after_delta,
                             delta_row_num):
        """
        This function insert each branch info to the relevant dict and DFs
        :param str branch_key: comment_id of the branch concat with _
        :param str submission_id: the submission ID of the branch
        :param branch_df: pandas DF: DF with the info of the branch: for each comment: comment_id, comment_depth, delta
        :param int num_delta: number of deltas in the branch
        :param int num_comments_after_delta: number of comments after the delta in the branch, or np.nan if no delta
        :param int delta_row_num: the index of the delta in the branch, or np.nan if no delta
        :return:
        """
        # add the branch to the hash tables
        self.branch_key_branch_id_hash_table[branch_key] = self.number_branches
        self.branch_id_branch_key_hash_table[self.number_branches] = branch_key
        # add to the branch df the submission id and the comment_real_depth (the index in the branch_df)
        branch_df = branch_df.assign(submission_id=submission_id)
        branch_df = branch_df.assign(comment_real_depth=branch_df.index)
        branch_df = branch_df.assign(branch_id=self.number_branches)
        self.branch_comments_info_df = self.branch_comments_info_df.append(branch_df, ignore_index=True)
        self.branch_info_dict[submission_id][branch_key] = [branch_df, branch_df.shape[0],
                                                            int(num_delta), num_comments_after_delta, delta_row_num]
        branch_row_info = pd.DataFrame({'submission_id': submission_id, 'branch_id': self.number_branches,
                                        'branch_key': branch_key, 'branch_length': branch_df.shape[0],
                                        'num_delta': int(num_delta),
                                        'num_comments_after_delta': num_comments_after_delta,
                                        'delta_index_in_branch': delta_row_num}, index=[self.number_branches])
        self.number_branches += 1
        if self.number_branches % 100 == 0:
            print('{}: Done {} branches'.format((time.asctime(time.localtime(time.time()))), self.number_branches))
        self.branch_numbers_df = self.branch_numbers_df.append(branch_row_info)

        return

    def find_branch_in_sub_tree(self, stack, node_children_df, delta_in_branch):
        """
        This is a recursive function that for each node append it to the stack.
        Then go over all its children and for each child - if it has children (not a leaf) - call the recursive function
        If not (it is a leaf): this is the end of a branch: insert the branch to the dict and pop the child from the stack
        If we go over all the children - pop the node from the stack and return
        :param list(tuple) stack: a list of tuples with info regarding the branch nodes
        :param node_children_df: pandas DF: a df with the node's children info
        :param bool delta_in_branch: if there is a delta in that branch
        :return:
        """
        # for each child: add to the stack and if it has children - call the recursive function
        for child_index, child in node_children_df.iterrows():
            # add the child to the stack
            stack.append((child['comment_id'], child['comment_depth'], child['delta']))
            # get the child's children
            child_children_df = self.submission_comments_df.loc[
                self.submission_comments_df['parent_id'] == child['comment_id']]
            # update delta_in_branch
            if child['delta'] == 1:  # the node got delta
                delta_in_branch = True
            if not child_children_df.empty:  # if it has children - call the function:
                self.find_branch_in_sub_tree(stack, child_children_df, delta_in_branch)
            else:  # there are no children - this is a leaf: add the branch to the dict
                branch_df = pd.DataFrame(stack, columns=['comment_id', 'comment_depth', 'delta'])
                num_delta = branch_df['delta'].sum()
                branch_key = branch_df['comment_id'].str.cat(sep='_')
                # get the number of comments after delta:
                delta_comment = branch_df.loc[branch_df['delta'] == 1]
                if num_delta == 1:  # only 1 delta in branch
                    delta_row_num = delta_comment.index + 1  # add 1 because the index starts from 0
                    num_comments_after_delta = branch_df.shape[0] - delta_row_num
                    num_comments_after_delta = int(num_comments_after_delta[0])
                elif num_delta > 1:  # more than 1 delta in the branch
                    print('more than 1 delta in branch', branch_key,
                          'save the number of comments after the last delta')
                    # get all the indexes of the delta in the branch
                    delta_row_list = list(delta_comment.index)
                    # get the last index
                    delta_row_num = delta_row_list[-1] + 1  # add 1 because the index starts from 0
                    num_comments_after_delta = branch_df.shape[0] - delta_row_num
                    num_comments_after_delta = int(num_comments_after_delta)
                    print('{}: branch {} has {} deltas'.format((time.asctime(time.localtime(time.time()))), branch_key,
                                                               num_delta))
                else:  # no delta in branch
                    delta_row_num = np.nan
                    num_comments_after_delta = np.nan
                # insert branch to dict and df
                self.insert_row_branch_df(branch_key, child['submission_id'], branch_df, num_delta,
                                          num_comments_after_delta, delta_row_num + 1)
                stack.pop()  # take out the leaf
                # update the "root parameters" before next branch (at the end of this branch):
                self.num_branches_in_root += 1
                self.num_deltas_in_root += int(num_delta)
                if num_delta > 0:  # if there are deltas in this branch
                    self.num_branches_with_delta_in_root += 1
                if num_comments_after_delta > 0:  # if there are comments after delta in this branch
                    self.num_branches_comments_after_delta_in_root += 1

        stack.pop()  # finish with the children of this node
        return

    def create_branch_info_from_data(self):
        """
        Create information for each branch
        For each submission, go over its roots and for each root call the recursive function find_branch_in_sub_tree
        that create branches for this root and insert their info to the dict.
        At the end of each root - insert its info to the dict
        :return:
        """
        # only comments that are roots
        only_roots = copy(self.comments_with_label.loc[self.comments_with_label['comment_is_root']])
        only_roots = only_roots[['comment_id', 'submission_id', 'comment_depth', 'delta']]

        # submission_id list:
        submission_id_list = only_roots['submission_id'].unique()

        # insert all comments of each submission to branch_info_dict
        for submission_id in submission_id_list:
            print('{}: Start create branches of submission_id {}. Submission number {}'.
                  format((time.asctime(time.localtime(time.time()))), submission_id, self.number_submissions))
            self.number_submissions += 1
            # get all submission's roots
            submission_roots_df = only_roots.loc[only_roots['submission_id'] == submission_id]
            # get all submission's comments that are not roots
            self.submission_comments_df = self.not_roots.loc[self.not_roots['submission_id'] == submission_id]
            # a stack for the algorithm
            stack = list()
            # for each root: create its branches
            for root_index, root in submission_roots_df.iterrows():
                # get the comment id of the root - without the _0
                root_id = root['comment_id']
                # run index of the number of branches with this root
                # get all the children of this root
                root_children_df = self.submission_comments_df.loc[self.submission_comments_df['parent_id'] == root_id]
                root_delta = root['delta']
                # get the root info: (comment_id, comment_depth, delta) if we create a new key:value
                stack.append((root['comment_id'], root['comment_depth'], root_delta))
                if root_children_df.empty:  # the root doesn't have a branch
                    # print('root', root_id, 'does not have a children')
                    # add the root without the children to both dicts
                    if root_delta == 1:
                        num_comments_after_delta = 0
                        delta_row_num = 1
                    else:
                        num_comments_after_delta = np.nan
                        delta_row_num = np.nan
                    # update the root's parameters:
                    self.num_branches_in_root = 1
                    self.num_deltas_in_root, self.num_branches_with_delta_in_root = root_delta, root_delta
                    self.num_branches_comments_after_delta_in_root = 0
                    # insert root info to dict and df
                    self.insert_row_root_df(submission_id, root_id)
                    # insert branch info to dict and df
                    branch_df = pd.DataFrame(stack, columns=['comment_id', 'comment_depth', 'delta'])
                    self.insert_row_branch_df(root_id, submission_id, branch_df, root_delta, num_comments_after_delta,
                                              delta_row_num)

                    # take the root out of the stack
                    stack.pop()
                else:
                    is_delta_root = bool(root_delta)
                    # call the recursive function for the root
                    self.find_branch_in_sub_tree(stack, root_children_df, is_delta_root)
                    # finish with this root:
                    # insert root info to the dict and df:
                    self.insert_row_root_df(submission_id, root_id)

        return

    def create_save_df(self):
        """
        This function create DFs from the dicts and save both dfs and dicts to pickle or csv
        :return:
        """
        # save dicts and dfs
        with open(os.path.join(data_directory, 'root_info_dict.pickle'), 'wb') as handle:
            pickle.dump(self.root_info_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(data_directory, 'branch_info_dict.pickle'), 'wb') as handle:
            pickle.dump(self.branch_info_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.branch_numbers_df.to_csv(os.path.join(data_directory, 'branch_numbers_df.csv'))
        self.branch_comments_info_df.to_csv(os.path.join(data_directory, 'branch_comments_info_df.csv'))
        self.root_info_df.to_csv(os.path.join(data_directory, 'root_info_df.csv'))
        # save hash tables
        branch_id_branch_key_hash_table_df = pd.DataFrame.from_dict(self.branch_id_branch_key_hash_table,
                                                                    orient='index')
        branch_id_branch_key_hash_table_df.columns = ['branch_key']
        branch_id_branch_key_hash_table_df.to_csv(os.path.join(data_directory, 'branch_id_branch_key_hash_table.csv'))
        branch_key_branch_id_hash_table_df = pd.DataFrame.from_dict(self.branch_key_branch_id_hash_table,
                                                                    orient='index')
        branch_key_branch_id_hash_table_df.columns = ['branch_id']
        branch_key_branch_id_hash_table_df.to_csv(os.path.join(data_directory, 'branch_key_branch_id_hash_table.csv'))

        # merge data with branch info
        data_with_branch_info = pd.merge(left=self.comments_with_label, right=self.branch_comments_info_df,
                                         on=['comment_id', 'comment_depth', 'submission_id', 'delta'], how='inner')
        data_with_branch_info.to_csv(os.path.join(
            data_directory, 'all_submissions_comments_with_label_all_deltalog_final_with_branches.csv'))

        # print numbers:
        print('{}: Finish running. \nTotal number of roots is: {}, total number of branches is: {} in {} submissions'.
              format((time.asctime(time.localtime(time.time()))), self.number_roots, self.number_branches,
                     self.number_submissions))

        return


def main():
    branch_obj = BranchStatistics()
    branch_obj.create_branch_info_from_data()
    branch_obj.create_save_df()


if __name__ == '__main__':
    main()
