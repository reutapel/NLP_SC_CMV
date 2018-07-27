import torch as tr
import torch.utils.data as dt
#import torchtext
#import spacy
import gensim


class CustomDataset(dt.Dataset):

    """
    Class for handling data before modeling, pre-process and loading
    """

    def __init__(self, branch_comments_raw_text_df, branch_comments_features_df, branch_comments_user_profiles_df,
                 branch_submission_dict, submission_data_dict, branch_deltas_data_dict, embeder):
        """
        this method uploads and organize all elements in the data for the different parts of the model
        """

        # IMPORTANT: all parts of data, are ordered by length of branch - descending, rest of entries are filled with zeros

        # First part of dataset is the raw text embedded vector, df M*N:
        # M - number of branches, N - maximum number of comments in branch, content - embedded vector of raw text
        self.branch_comments_raw_text_df = branch_comments_raw_text_df

        # Second part of dataset is the features of the comments, df M*N, content - comment features
        self.branch_comments_features_df = branch_comments_features_df

        # Third part of dataset is the profiles of the users, df M*N, content - user's features
        self.branch_comments_user_profiles_df = branch_comments_user_profiles_df

        # Forth part of dataset contains two dictionaries:
        # 1. {branch index: submission id}
        # 2. {submission id: [submission text, submission features, submitter profile features]}
        self.branch_submission_dict = branch_submission_dict
        self.submission_data_dict = submission_data_dict

        # Fifth part of the dataset is a dictionary: {branch index: [is delta in branch, number of deltas in branch,
        # [deltas comments location in branch]]}
        self.branch_deltas_data_dict = branch_deltas_data_dict

        self.embeder = embeder

        pass

    def __getitem__(self, index):
        """
        this method takes all elements of a single data point for one iteration of the model, and returns them with
        it's label. the dataloader will use this method for taking batches in training.
        :param index: index of data point to be retrieved
        :return: (data point elements, label)
        """
        # TODO: embed comments in branch_comments_raw_text_df , embed submission text
        X = [self.branch_comments_raw_text_df.iloc[index],self.branch_comments_features_df[index],
             self.branch_comments_user_profiles_df[index], self.submission_data_dict[self.branch_submission_dict[index]]]
        y = self.branch_deltas_data_dict[index]
        return (X, y)

    def __len__(self):
        """

        :return: size of dataset
        """

        return len(self.branch_comments_raw_text_df.index)
# TODO: understand dimensions of df vs. new tensor
# TODO: build function that embeds the raw text and put the function in a different module
    def df_to_tensor(self, df):
        """
        this method takes a df and returns a tensor of embedded / numerical raw data ready for model input
        :return: tensor
        """
        seq_length = len(df.columns)

        for row in df.iteritems():
            hello = 1


        return