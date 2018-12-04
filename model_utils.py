import torch as tr
import torch.utils.data as dt
#import torchtext
#import spacy
#import gensim
import torch.nn.functional as F
import numbers
import numpy as np


class CustomDataset(dt.Dataset):

    """
    Class for handling data before modeling, pre-process and loading
    """
    # TODO: REUT: df columns should be com1, com2 ... important for tensor ordering
    def __init__(self, branch_comments_embedded_text_df, branch_comments_features_df, branch_comments_user_profiles_df,
                 branch_submission_dict, submission_data_dict, branch_deltas_data_dict, branches_lengths_list):
        """
        this method uploads and organize all elements in the data for the different parts of the model
        all parts of data, are ordered by length of branch - descending, rest of entries are filled with zeros
        :param branch_comments_embedded_text_df: raw text embedded, df M*N, M - number of branches, N - maximum number
        of comments in branch, content - embedded vectors of raw text
        :param branch_comments_features_df: features of the comments, df M*N, content - comment features
        :param branch_comments_user_profiles_df: profiles of the users, df M*N, content - user's features
        :param branch_submission_dict: {branch index: [submission id, [branch features]]}
        :param submission_data_dict: {submission id: [submission text, submission features, submitter profile features]}
        :param branch_deltas_data_dict: {branch index:
        [is delta in branch, number of deltas in branch, [deltas comments location in branch]]}
        :param branches_lengths_list: actual branches lengths without padding for the batch learning
        """

        self.branch_comments_embedded_text_tensor = self.df_to_tensor(branch_comments_embedded_text_df)

        self.branch_comments_features_tensor = self.df_to_tensor(branch_comments_features_df)

        self.branch_comments_user_profiles_tensor = self.df_to_tensor(branch_comments_user_profiles_df)

        self.branch_submission_dict = branch_submission_dict
        self.submission_data_dict = submission_data_dict

        self.branch_deltas_data_dict = branch_deltas_data_dict

        self.branches_lengths = branches_lengths_list

    def __getitem__(self, index):
        """
        this method takes all elements of a single data point for one iteration of the model, and returns them with
        it's label. the dataloader will use this method for taking batches in training.
        :param index: index of data point to be retrieved
        :return: (data point elements, label)
        """
        # if type(self.submission_data_dict[self.branch_submission_dict[index][0]][1]) == np.ndarray
        x = [self.branch_comments_embedded_text_tensor[index],
             self.branch_comments_features_tensor[index],
             self.branch_comments_user_profiles_tensor[index],
             tr.Tensor(self.submission_data_dict[self.branch_submission_dict[index][0]][0]),
             tr.Tensor(self.submission_data_dict[self.branch_submission_dict[index][0]][1]),
             tr.Tensor(self.submission_data_dict[self.branch_submission_dict[index][0]][2]),
             tr.Tensor(self.branch_submission_dict[index][1].astype('float')),
             self.branches_lengths[index]]
        y = self.branch_deltas_data_dict[index][0]
        return x, y

    def __len__(self):
        """

        :return: size of dataset
        """

        return self.branch_comments_embedded_text_tensor.size()[0]

    def df_to_tensor(self, df):
        """
        this method takes a df of values / vectors and returns a tensor of 2 dim/ 3 dim accordingly
        :return: tensor
        """

        # get shapes
        df_rows_num = df.shape[0]
        df_columns_num = df.shape[1]

        # if values of df is numbers
        if isinstance(df.iloc[0, 0], numbers.Number):
            # print("new tensor shape is", df_rows_num, ",", df_columns_num)
            return tr.Tensor(df.values)
        # if values of df are vectors
        else:
            df_value_length = len(df.iloc[0, 0])
            tensor = tr.Tensor([[column for column in row] for row in df.values])
            # print("new tensor shape is", df_rows_num, ",", df_columns_num, ",", df_value_length)

            return tensor


class InitLstm:
    def __init__(self, input_size, hidden_size, num_layers, batch_first=True):
        """
        initializes LSTM parameters
        :param input_size: input size for LSTM
        :param hidden_size: output size vector of LSTM
        :param num_layers: number of layers in LSTM
        :param batch_first: if true tensor first dimension is batch size of LSTM
        """

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first


class InitConv1d:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, leaky_relu_alpha):
        """
        initializes conv1d layer parameters
        :param in_channels(int): Number of channels in the input
        :param out_channels(int): Number of channels produced by the convolution
        :param kernel_size(int or tuple): Size of the convolving kernel
        :param stride(int or tuple, optional): Stride of the convolution. Default: 1
        :param padding(int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        :param leaky_relu_alpha: value of breaking linearity for this convolution
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.leakyRelu = leaky_relu_alpha