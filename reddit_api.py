import praw
import time
import csv
import pickle
import logging
import os
from datetime import datetime
import pandas as pd
from collections import defaultdict
import re
from copy import copy
import numpy as np

# configurate logging
base_directory = os.path.abspath(os.curdir)
log_directory = os.path.join(base_directory, 'logs')
results_directory = os.path.join(base_directory, 'change my view')
LOG_FILENAME = os.path.join(log_directory,
                            datetime.now().strftime('LogFile_importing_change_my_view_%d_%m_%Y_%H_%M_%S.log'))
logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO, )


class ApiConnection:

    def __init__(self, subreddit):

        # configurate API connection details
        user_agent = "learning API 1.0"
        client_id = 'learning API 1.0'
        user = 'ssheiba'
        password = 'Angui100'
        ssheiba_client_id = 'ysLnQ_KXpkYA1g'
        ssheiba_client_secret = 'pLMhOt7sVy6IHdx5mcCy48yB6Ow'
        ssheiba_user_agent = 'learningapp:com.example.myredditapp:v1.2.3 (by /u/ssheiba)'
        self.r_connection = praw.Reddit(client_id=ssheiba_client_id, user_agent=ssheiba_user_agent,
                                        client_secret=ssheiba_client_secret, username=user,password=password)
        self.subreddit_name = subreddit

        return

    # def get_karma(self, user_id):
    #     user = self.r_connection.redditor(user_id)
    #     comments = user.comments
    #     gen = user.get_submitted(limit=10)
    #     karma_by_subreddit = {}
    #     for thing in gen:
    #         subreddit = thing.subreddit.display_name
    #         karma_by_subreddit[subreddit] = (karma_by_subreddit.get(subreddit, 0)
    #                                          + thing.score)

    def get_submissions(self):

        num_of_total_submissions = 0
        submissions = list()
        with open(os.path.join(results_directory, 'all submissions.csv'), 'a') as file:
            writer = csv.writer(file, lineterminator='\n')
            fieldnames = ['submission_author', 'submission_title', 'submission_comments_by_id',
                          'submission_created_utc', 'submission_edited', 'submission_body', 'submission_id',
                          'submission_likes', 'submission_ups', 'submission_downs', 'submission_score',
                          'submission_num_reports', 'submission_gilded', 'submission_distinguished',
                          'submission_is_crosspostable', 'submission_banned_by', 'submission_banned_at_utc',
                          'submission_removal_reason', 'submission_clicked', 'submission_num_comments',
                          'submission_contest_mode', 'submission_media']
            writer.writerow(fieldnames)
        subids = set()
        for submission in self.r_connection.subreddit(self.subreddit_name).submissions():
            with open(os.path.join(results_directory, 'all submissions.csv'), 'a') as file:
                writer = csv.writer(file, lineterminator='\n')
                writer.writerow([submission.author, submission.title.encode('utf-8'),
                                 submission._comments_by_id, submission.created_utc,
                                 submission.edited, submission.selftext.encode('utf-8'), submission.id,
                                 submission.likes, submission.ups, submission.downs, submission.score,
                                 submission.num_reports, submission.gilded, submission.distinguished,
                                 submission.is_crosspostable, submission.banned_by, submission.banned_at_utc,
                                 submission.removal_reason, submission.clicked, submission.num_comments,
                                 submission.contest_mode, submission.media])

            subids.add(submission.id)
            submissions.append(submission)
            num_of_total_submissions += 1
            print("added submission id : {}".format(submission.id))
            print("total number of submissions so far is {}".format(num_of_total_submissions))
            logging.info("added submission id : {}".format(submission.id))
            logging.info("total number of submissions so far is {}".format(num_of_total_submissions))
        subid = list(subids)
        print("all subids are: {}".format(subid))
        logging.info("all subids are: {}".format(subid))
        # save all submissions object
        print("saving submissions list")
        logging.info("saving submissions list")
        with open('submissions.pickle', 'wb') as handle:
            pickle.dump(submissions, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return subid

    def parse_comments(self, subid, filename):
        """
        this method retrieves all comments for the subid list (and their replies)
        :param subid: list of submissions id to get their comment tree
        :param filename: name for dataframe to file to be saved, dynamic for different calls to this method
        :return: saves features of the comments and replies, among with the data objects of them.
        """
        comments = dict()
        num_of_total_comments = 0
        index = len(subid)

        # prepare the file
        with open(os.path.join(results_directory, filename), 'a') as file:
            writer = csv.writer(file, lineterminator='\n')
            fieldnames2 = ['comment_author', 'comment_created_utc', 'comment_edited', 'comment_body',
                           'comment_path', 'comment_id', 'parent_id', 'submission_id', 'comment_is_submitter',
                           'comment_likes', 'comment_is_root', 'comment_ups', 'comment_downs', 'comment_score',
                           'comment_num_reports', 'comment_gilded', 'comment_distinguished', 'comment_controversiality',
                           'comment_banned_by', 'comment_banned_at_utc', 'comment_depth', 'comment_removal_reason']
            writer.writerow(fieldnames2)

        # iterate all submissions
        for i in range(0, index):
            print('{}: start submission {} for id {}'.format((time.asctime(time.localtime(time.time()))), i, subid[i]))
            logging.info('{}: start submission {} for id {}'.format((time.asctime(time.localtime(time.time()))), i,
                                                                    subid[i]))
            submission = self.r_connection.submission(id=subid[i])
            print('{}: start more comments'.format((time.asctime(time.localtime(time.time())))))
            logging.info('{}: start more comments'.format((time.asctime(time.localtime(time.time())))))

            # replace the more comments objects with the comments themselves (flatten trees of replies of comments)
            submission.comment_sort = 'new'
            submission.comments.replace_more(limit=None)

            comments[subid[i]] = list()

            # retrieve all comments data
            for comment in submission.comments.list():

                comments[subid[i]].append(comment)
                with open(os.path.join(results_directory, filename), 'a') as file:
                    writer = csv.writer(file, lineterminator='\n')
                    writer.writerow([comment.author, comment.created_utc,
                                     comment.edited,
                                     comment.body.encode('utf-8'), comment.permalink.encode('utf-8'),
                                     comment.id.encode('utf-8'), comment.parent_id.encode('utf-8'),
                                     comment.submission.id.encode('utf-8'),comment.is_submitter,
                                     comment.likes, comment.is_root, comment.ups, comment.downs, comment.score,
                                     comment.num_reports, comment.gilded, comment.distinguished,
                                     comment.controversiality, comment.banned_by, comment.banned_at_utc, comment.depth,
                                     comment.removal_reason])
            print("number of comments for subid {} is {}".format(subid[i], len(comments[subid[i]])))
            logging.info("number of comments for subid {} is {}".format(subid[i], len(comments[subid[i]])))
            num_of_total_comments += len(comments[subid[i]])
            print("total number of comments so far is {}".format(num_of_total_comments))
            logging.info("total number of comments so far is {}".format(num_of_total_comments))

        # extract deltas from comments
        all_submissions_comments = pd.read_csv(os.path.join(results_directory, filename))
        OP_deltas_comments_ids = self.get_deltas_manual(all_submissions_comments)

        return all_submissions_comments, OP_deltas_comments_ids

    def get_deltas_manual(self, all_submissions_comments):

        delta_comments_depth_zero = pd.DataFrame(columns=['comment_id', 'parent_id'])
        # OP_deltas_comments_ids: {submission_id: [comments that gor delta]}
        OP_deltas_comments_ids = defaultdict(list)
        delta_tokens = ['&amp;#8710;', '&#8710;', '&#916;', '&amp;916;', '∆', '!delta', 'Δ', '&delta;']
        num_of_deltas = 0
        # OP_deltas: {(submission_id, comment_that_got_delta_is) : {comment_gave_delta_id+desc.: value}}
        OP_deltas = defaultdict(dict)

        # find all delta comments and save their details in OP_deltas_comments_ids and the comments that got the delta
        # ID in OP_deltas
        for index, row in all_submissions_comments.iterrows():

            # check if legal delta
            if row.loc['comment_is_submitter'] == True and any(delta_token in row.loc['comment_body'] for delta_token
                                                               in delta_tokens) \
                    and len(row.loc['comment_body']) > 50:

                # if delta's parent is submission:
                # comment_depth = 0 is the first comment in the tree
                if row.loc['comment_depth'] == 0:
                    delta_comments_depth_zero.append([row.loc['comment_id'], row.loc['parent_id']])
                    print("comment's parent is submission")
                    continue

                # check that OP is not giving a delta to himself or to the deltabot
                parent_id = row.loc['parent_id']
                parent_id = parent_id.replace("b't1_", "").replace("'", "")
                # the comment that is the parent of row (got the delta)
                delta_comment = all_submissions_comments.loc[all_submissions_comments["comment_id"]
                                                             .str.lstrip("b").str.strip("'") == parent_id]

                # if comment of parent_id is not in data
                if delta_comment.empty:
                    continue
                # check if the author of the parent of row is not the submitter
                check_delta = copy(delta_comment.loc[delta_comment['comment_is_submitter'] == True])

                if not check_delta.empty:
                    print("submitter gave delta to himself")
                # comments that are the row's parent and not written by the submitter or delta bot
                real_delta = copy(delta_comment.loc[(delta_comment['comment_is_submitter'] == False)
                                  & (delta_comment['comment_author'] != "DeltaBot")])
                if not real_delta.empty:
                    num_of_deltas += 1
                    if num_of_deltas % 100 == 0:
                        print("{} deltas".format(num_of_deltas))
                    OP_deltas_comments_ids[row.submission_id].append(row.parent_id)
                    OP_deltas[(row.submission_id, row.parent_id)][row.comment_id + "_" + "delta_OP"] = row.comment_author
                    OP_deltas[(row.submission_id, row.parent_id)][
                        row.comment_id + "_" + "delta_date"] = row.comment_created_utc
                    OP_deltas[(row.submission_id, row.parent_id)][row.comment_id + "_" + "delta_body"] = row.comment_body
                    OP_deltas[(row.submission_id, row.parent_id)][row.comment_id + "_" + "delta_path"] = row.comment_path
                    OP_deltas[(row.submission_id, row.parent_id)][row.comment_id + "_" + "delta_id"] = row.comment_id
                    OP_deltas[(row.submission_id, row.parent_id)][row.comment_id + "_" + "delta_parent_id"] = row.parent_id
                    OP_deltas[(row.submission_id, row.parent_id)][
                        row.comment_id + "_" + "delta_submission_id"] = row.submission_id

        # save delta data
        print("save delta data")
        logging.info("save delta data")
        with open('OP_deltas_comments_ids.pickle', 'wb') as handle:
            pickle.dump(OP_deltas_comments_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('OP_deltas.pickle', 'wb') as handle:
            pickle.dump(OP_deltas, handle, protocol=pickle.HIGHEST_PROTOCOL)
        delta_comments_depth_zero.to_csv(os.path.join(results_directory, 'delta_comments_depth_zero.csv'))
        return OP_deltas_comments_ids

    def get_deltas_log(self, delta_log):
        """
        this method retrieves all deltas awarded in CMV and saves that log to a csv.
        :param delta_log: name of subreddit that holds all the deltas given in CMV subreddit
        :return:
        """
        num_of_total_deltas = 0

        # create file
        with open(os.path.join(results_directory, 'all deltas.csv'), 'a') as file:
            writer = csv.writer(file, lineterminator='\n')
            fieldnames = ['delta_author', 'delta_title', 'delta_created_utc', 'delta_selftext', 'delta_id',
                          'delta_likes', 'delta_ups', 'delta_downs', 'delta_score', 'delta_name', 'delta_permalink']
            writer.writerow(fieldnames)

        # write all deltas to file
        for delta in self.r_connection.subreddit(delta_log).submissions():
            with open(os.path.join(results_directory, 'all deltas.csv'), 'a') as file:
                writer = csv.writer(file, lineterminator='\n')
                writer.writerow([delta.author, delta.title.encode('utf-8'), delta.created_utc,
                                 delta.selftext.encode('utf-8'), delta.id.encode('utf-8'),delta.likes, delta.ups,
                                 delta.downs, delta.score, delta.name, delta.permalink.encode('utf-8')])
            num_of_total_deltas += 1
            print("added delta id : {} of title: {}".format(delta.id, delta.title))
            print("total number of deltas so far is {}".format(num_of_total_deltas))
            logging.info("added delta id : {} of title: {}".format(delta.id, delta.title))
            logging.info("total number of deltas so far is {}".format(num_of_total_deltas))

        # parse delta logs for OP deltas
        deltas = pd.read_csv(os.path.join(results_directory, 'all deltas.csv'), index_col=False)

        OP_deltas_comments_ids_deltalog = self.parse_op_deltas(deltas)

        return OP_deltas_comments_ids_deltalog

    def parse_op_deltas(self, deltas):
        """
        this method parse the text of each delta comment from delta log and saves the IDs of the ones given by OP
        :return: OP delta comment ids dict {submission id: [comments id]}
        """

        OP_deltas_comments_ids_deltalog = defaultdict(list)
        comments_with_delta_ids = list()
        delta_count = 0
        for df_index, row in deltas.iterrows():

            # obtain submissionid
            submission_id_indexes = [m.start() for m in
                                     re.finditer('/comments/', deltas.loc[df_index, "delta_selftext"])]
            if len(submission_id_indexes) < 1:
                print("bug")
            submission_id_text = deltas.loc[df_index, "delta_selftext"][:submission_id_indexes[0] + 20]
            submission_id_text_parsed = submission_id_text.split("/")
            comments_idx = submission_id_text_parsed.index("comments")
            submission_id = submission_id_text_parsed[comments_idx + 1]

            # obtain OP user name
            op_username_indexes = [m.start() for m in
                                   re.finditer('"opUsername":', deltas.loc[df_index, "delta_selftext"])]
            op_username_text = deltas.loc[df_index, "delta_selftext"][op_username_indexes[0]:]
            left_op_username_text = op_username_text.partition("\\n")[0]
            op_username = left_op_username_text.partition('",\\n')[0]
            op_username = op_username.split(": ")[1]
            op_username = op_username.strip(",'")

            # parse from delta text all comments id's that got deltas.
            deltas_indexes = [m.start() for m in re.finditer('"awardedLink":', deltas.loc[df_index, "delta_selftext"])]
            delta_awarding_indexes = [m.start() for m in
                                      re.finditer('"awardingUsername":', deltas.loc[df_index, "delta_selftext"])]

            if len(deltas_indexes) != len(delta_awarding_indexes):
                print("match error")
                break

            for delta_list_idx, delta_index in enumerate(deltas_indexes):

                # get & check if awarding is OP
                awarding_text = deltas.loc[df_index, "delta_selftext"][delta_awarding_indexes[delta_list_idx]:]
                left_awarding_text = awarding_text.partition("\\n")[0]
                awarding_username = left_awarding_text.partition('",\\n')[0]
                awarding_username = awarding_username.split(": ")[1]
                awarding_username = awarding_username.strip(",'")

                if awarding_username != op_username:
                    print("not OP's delta")
                    continue

                # get awarded delta comment id index
                text = deltas.loc[df_index, "delta_selftext"][delta_index:]
                left_text = text.partition("\\n")[0]
                splited = text.split('/')
                for idx, parts in enumerate(splited):
                    if "cmv_" in parts:
                        comment_id = splited[idx + 1].partition('",\\n')[0]
                        print(comment_id)
                        delta_count += 1
                        print(delta_count)
                        comments_with_delta_ids.append(comment_id)
                        OP_deltas_comments_ids_deltalog[submission_id].append(comment_id)

        comments_with_delta_ids = list(set(comments_with_delta_ids))
        print("num of deltas from deltalog is: {}".format(len(comments_with_delta_ids)))

        # save delta data
        print("save delta log data ")
        logging.info("save delta log data")
        with open('OP_deltas_comments_ids_deltalog.pickle', 'wb') as handle:
            pickle.dump(OP_deltas_comments_ids_deltalog, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('comments_with_delta_ids.pickle', 'wb') as handle:
            pickle.dump(comments_with_delta_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return OP_deltas_comments_ids_deltalog

    def create_label(self, comments, OP_deltas_comments_ids_deltalog, OP_deltas_comments_ids, file_name):
        """
        this method creates a label for delta comments both by the delta log and the manual parsing of the data
        :param comments: the data
        :param OP_deltas_comments_ids_deltalog: dict of submission_id : comment id from deltalog
        :param OP_deltas_comments_ids: dict of submission_id : comment id from manual extraction
        :param file_name: dynamic name for new data with label to be saved
        :return:
        """

        # create label column
        comments["delta"] = 0

        # for each comment in data , check if it's in one of the deltas dict list by it's submissionID
        for index, row in comments.iterrows():

            if type(row.loc['comment_id']) is float:
                print("not real comment id : {}".format(row.loc['comment_id']))
                continue

            deltalog_comments = list()
            manual_comments = list()

            # get delta comments of this submissionid
            try:
                deltalog_comments = OP_deltas_comments_ids_deltalog[row.loc['submission_id'].lstrip("b").strip("'")]
            except ValueError:
                print("no delta comments for submission: {} in deltalog".format(row.loc['submission_id']))

            try:
                manual_comments = OP_deltas_comments_ids[row.loc['submission_id']]
                manual_comments = [w.lstrip("b") for w in manual_comments]
                manual_comments = [w.strip("'") for w in manual_comments]
                manual_comments = [w.replace("t1_", "") for w in manual_comments]

            except ValueError:
                print("no delta comments for submission: {} in manual".format(row.loc['submission_id']))

            # check if this comment got delta
            if row.loc['comment_id'].lstrip("b").strip("'") in deltalog_comments \
                    or row.loc['comment_id'].lstrip("b").strip("'") in manual_comments:
                comments.loc[index, "delta"] = 1

        # save data with label
        comments.to_csv(file_name)
        return comments

    def complete_deltas_from_log_not_in_data(self,OP_deltas_comments_ids_deltalog, all_submissions_comments_with_label):
        """
        this method checks which submissions appeared in delta log and not in imported data and brings the missing data,
        then concatenate with existing data
        :param OP_deltas_comments_ids_deltalog: dictionary of submissions that have delta comments from delta log
        :param all_submissions_comments_with_label: imported comments data
        :return: complete data
        """

        # get submissions from delta log that are not in data
        deltas_log_intersection_keys = set(
            all_submissions_comments_with_label["submission_id"].str.lstrip("b").str.strip("'")).intersection(
            set(OP_deltas_comments_ids_deltalog.keys()))
        log_keys_not_in_data = list(
            set(OP_deltas_comments_ids_deltalog.keys()).difference(deltas_log_intersection_keys))

        # save them
        with open('submissions_from_delta_log_not_in_data.pickle', 'wb') as handle:
            pickle.dump(log_keys_not_in_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # get submissions and comments of submission from delta log that are not in data
        filename = 'all_submissions_comments_of_submission_deltalog_not_in_data.csv'
        all_submissions_comments_deltalog_not_in_data, OP_deltas_comments_ids  = \
            self.parse_comments(log_keys_not_in_data, filename)

        # create label only for new data
        file_name = "all_submissions_comments_deltalog_not_in_data_with_label.csv"
        all_submissions_comments_deltalog_not_in_data_with_label = \
            self.create_label(all_submissions_comments_deltalog_not_in_data, OP_deltas_comments_ids_deltalog,
                          OP_deltas_comments_ids, file_name)

        # concatenate old and new data
        all_submissions_comments_label_union = all_submissions_comments_with_label.append(
            all_submissions_comments_deltalog_not_in_data_with_label)

        all_submissions_comments_label_union.to_csv("all_submissions_comments_with_label_all_deltalog_final.csv")

        return all_submissions_comments_label_union

    def get_all_submissions_final_data(self):

        all_submissions_comments_label_union = pd.read_csv("all_submissions_comments_with_label_all_deltalog_final.csv")
        all_subids = list(all_submissions_comments_label_union.submission_id.unique())

        num_of_total_submissions = 0
        submissions = list()
        with open(os.path.join(results_directory, 'all_submissions_final.csv'), 'a') as file:
            writer = csv.writer(file, lineterminator='\n')
            fieldnames = ['submission_author', 'submission_title', 'submission_created_utc', 'submission_body',
                          'submission_id']
            writer.writerow(fieldnames)
        subids = set()
        all_subids = [w.lstrip("b") for w in all_subids]
        all_subids = [w.strip("'") for w in all_subids]
        for subid in all_subids:
            submission = self.r_connection.submission(id=subid)
            with open(os.path.join(results_directory, 'all_submissions_final.csv'), 'a') as file:
                writer = csv.writer(file, lineterminator='\n')
                writer.writerow([submission.author, submission.title.encode('utf-8'), submission.created_utc,
                                 submission.selftext.encode('utf-8'), submission.id])

            subids.add(submission.id)
            submissions.append(submission)
            num_of_total_submissions += 1
            print("added submission id : {}".format(submission.id))
            print("total number of submissions so far is {}".format(num_of_total_submissions))
            logging.info("added submission id : {}".format(submission.id))
            logging.info("total number of submissions so far is {}".format(num_of_total_submissions))
        subid = list(subids)
        print("all subids are: {}".format(subid))
        logging.info("all subids are: {}".format(subid))
        # save all submissions object
        print("saving submissions list")
        logging.info("saving submissions list")
        with open('submissions_final.pickle', 'wb') as handle:
            pickle.dump(submissions, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return


def main():
    subreddit = 'changemyview'
    print('{} : Run for sub reddit {}'.format((time.asctime(time.localtime(time.time()))), subreddit))
    logging.info('{} : Run for sub reddit {}'.format((time.asctime(time.localtime(time.time()))), subreddit))
    # create class instance
    connect = ApiConnection(subreddit)

    #get submissions of sub reddit
    subids = connect.get_submissions()

    #get comments of submissions
    data_name = "all_submissions_comments.csv"
    all_submissions_comments, OP_deltas_comments_ids = connect.parse_comments(subids, data_name)

    print('{} : finished Run for sub reddit {}'.format((time.asctime(time.localtime(time.time()))), subreddit))
    logging.info('{} : finished Run for sub reddit {}'.format((time.asctime(time.localtime(time.time()))), subreddit))

    # get outcome from delta log
    delta_log = 'DeltaLog'
    OP_deltas_comments_ids_deltalog = connect.get_deltas_log(delta_log)

    #create label
    df_name = "all_submissions_comments_with_label.csv"
    all_submissions_comments_with_label = connect.create_label(all_submissions_comments,
                                                               OP_deltas_comments_ids_deltalog, OP_deltas_comments_ids,
                                                               df_name)
    # add missing delta from delta log
    connect.complete_deltas_from_log_not_in_data(OP_deltas_comments_ids_deltalog, all_submissions_comments_with_label)

    # get all submissions data final
    #connect.get_all_submissions_final_data()


if __name__ == '__main__':
    main()
