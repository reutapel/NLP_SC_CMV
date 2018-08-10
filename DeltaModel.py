import torch as tr
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import math


class DeltaModel(nn.Module):
    """
    A deep learning model class with the following logic of taking 3 different sectors of signals in data for predicting
    if the event "delta" occurred in the discussion "branch"
    """
    def __init__(self, input_size_text, hidden_size_text, num_layers_text, batch_first_text,
                 input_size_comments, hidden_size_comments, num_layers_comments, batch_first_comments,
                 input_size_users, hidden_size_users, num_layers_users, batch_first_users, in_channels_text,
                 out_channels_text, kernel_size_text, stride_text, padding_text, in_channels_sub_features,
                 out_channels_sub_features, kernel_size_sub_features, stride_sub_features, padding_sub_features,
                 in_channels_sub_profile_features, out_channels_sub_profile_features, kernel_size_sub_profile_features,
                 stride_sub_profile_features, padding_sub_profile_features, input_size_text_sub, input_size_sub_features,
                 input_size_sub_profile_features, num_labels, batch_size, first_linear_reduction, second_linear_reduction):
        """

        :param input_size_text: input size for LSTM of embedded text
        :param hidden_size_text: output size vector of LSTM of embedded text
        :param num_layers_text: number of layers in LSTM of embedded text
        :param batch_first_text: if true tensor first dimension is batch size of LSTM of embedded text
        :param input_size_comments: input size for LSTM of comments features
        :param hidden_size_comments: output size vector of LSTM of comments features
        :param num_layers_comments: number of layers in LSTM of comments features
        :param batch_first_comments: if true tensor first dimension is batch size of LSTM of comments features
        :param input_size_users: input size for LSTM of user features
        :param hidden_size_users: output size vector of LSTM of user features
        :param num_layers_users: number of layers in LSTM of user features
        :param batch_first_users: if true tensor first dimension is batch size of LSTM of user features
        :param in_channels_text (int): Number of channels in the input
        :param out_channels_text (int): Number of channels produced by the convolution
        :param kernel_size_text (int or tuple): Size of the convolving kernel
        :param stride_text (int or tuple, optional): Stride of the convolution. Default: 1
        :param padding_text (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        :param in_channels_sub_features (int): Number of channels in the input
        :param out_channels_sub_features (int): Number of channels produced by the convolution
        :param kernel_size_sub_features (int or tuple): Size of the convolving kernel
        :param stride_sub_features (int or tuple, optional): Stride of the convolution. Default: 1
        :param padding_sub_features (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        :param in_channels_sub_profile_features (int): Number of channels in the input
        :param out_channels_sub_profile_features (int): Number of channels produced by the convolution
        :param kernel_size_sub_profile_features (int or tuple): Size of the convolving kernel
        :param stride_sub_profile_features (int or tuple, optional): Stride of the convolution. Default: 1
        :param padding_sub_profile_features (int or tuple, optional): Zero-padding added to both sides of the input.
        Default: 0
        :param input_size_text_sub: length of embedded text vector of submission
        :param input_size_sub_features: length of submission feature vector
        :param input_size_sub_profile_features: length of submitter feature vector
        :param num_labels: number of labels
        :param batch_size: batch size
        :param first_linear_reduction: linear size from branch hidden rep output to this size
        :param second_linear_reduction: linear size from first_linear_size output to this size
        """
        super(DeltaModel, self).__init__()

        self.hparams.on_gpu = tr.cuda.is_available()

        # define lstm's parameters
        self.batch_size = batch_size
        self.input_size_text = input_size_text
        self.hidden_size_text = hidden_size_text
        self.num_layers_text = num_layers_text
        self.batch_first_text = batch_first_text
        self.input_size_comments = input_size_comments
        self.hidden_size_comments = hidden_size_comments
        self.num_layers_comments = num_layers_comments
        self.batch_first_comments = batch_first_comments
        self.input_size_users = input_size_users
        self.hidden_size_users = hidden_size_users
        self.num_layers_users = num_layers_users
        self.batch_first_users = batch_first_users

        # initialize LSTM's hidden states
        self.hidden_text = self.init_hidden(self.num_layers_text, self.batch_size, self.hidden_size_text)
        self.hidden_users = self.init_hidden(self.num_layers_users, self.batch_size, self.hidden_size_comments)
        self.hidden_comments = self.init_hidden(self.num_layers_comments, self.batch_size, self.hidden_size_users)

        # define convolution's parameters
        self.in_channels_text = in_channels_text
        self.out_channels_text = out_channels_text
        self.kernel_size_text = kernel_size_text
        self.stride_text = stride_text
        self.padding_text = padding_text
        self.in_channels_sub_features = in_channels_sub_features
        self.out_channels_sub_features = out_channels_sub_features
        self.kernel_size_sub_features = kernel_size_sub_features
        self.stride_sub_features = stride_sub_features
        self.padding_sub_features = padding_sub_features
        self.in_channels_sub_profile_features = in_channels_sub_profile_features
        self.out_channels_sub_profile_features = out_channels_sub_profile_features
        self.kernel_size_sub_profile_features = kernel_size_sub_profile_features
        self.stride_sub_profile_features = stride_sub_profile_features
        self.padding_sub_profile_features = padding_sub_profile_features
        self.input_size_text_sub = input_size_text_sub
        self.input_size_sub_features = input_size_sub_features
        self.input_size_sub_profile_features = input_size_sub_profile_features

        # define layers
        # static layers definition- aggregates the parameters for the derivatives

        # LSTM layers
        # parameters -
        # input_size: embedding dimension or feature vector dimension
        # hidden_size: The number of features in the hidden state `h`
        # num_layers: number of lstm layers
        # batch_first: True if batch dimension is first

        self.lstm_text = nn.LSTM(
            input_size=self.input_size_text,
            hidden_size=self.hidden_size_text,
            num_layers=self.num_layers_text,
            batch_first=self.batch_first_text)

        self.lstm_comments = nn.LSTM(
            input_size=self.input_size_comments,
            hidden_size=self.hidden_size_comments,
            num_layers=self.num_layers_comments,
            batch_first=self.batch_first_comments)

        self.lstm_users = nn.LSTM(
            input_size=self.input_size_users,
            hidden_size=self.hidden_size_users,
            num_layers=self.num_layers_users,
            batch_first=self.batch_first_users)

        # convolution layers
        # parameters -
        # in_channels: number of channels in the input
        # out_channels: number of channels produced by the convolution (number of filters / kernels)
        # kernel_size: sliding window dimension
        # stride: step size of kernel
        # padding: zero padding added to both sides of the input

        self.conv_sub_text = nn.Conv1d(
            self.in_channels_text,
            self.out_channels_text,
            self.kernel_size_text,
            stride=self.stride_text,
            padding=self.padding_text)

        self.conv_sub_features = nn.Conv1d(
            self.in_channels_sub_features,
            self.out_channels_sub_features,
            self.kernel_size_sub_features,
            stride=self.stride_sub_features,
            padding=self.padding_sub_features)

        self.conv_sub_user = nn.Conv1d(
            self.in_channels_sub_profile_features,
            self.out_channels_sub_profile_features,
            self.kernel_size_sub_profile_features,
            stride=self.stride_sub_profile_features,
            padding=self.padding_sub_profile_features)

        self.leaky_relu_text = nn.LeakyReLU()
        self.leaky_relu_features = nn.LeakyReLU()
        self.leaky_relu_user = nn.LeakyReLU()

        _, _, output_length_text = self.calc_conv1d_output_shape(self.batch_size, self.out_channels_text,
                                                                 self.input_size_text_sub ,self.padding_text,
                                                                 self.kernel_size_text, self.stride_text)
        _, _, output_length_comments = self.calc_conv1d_output_shape(self.batch_size, self.out_channels_sub_features,
                                                                     self.input_size_sub_features,
                                                                     self.padding_sub_features,
                                                                     self.kernel_size_sub_features,
                                                                     self.stride_sub_features)
        _, _, output_length_users = self.calc_conv1d_output_shape(self.batch_size,
                                                                  self.out_channels_sub_profile_features,
                                                                  self.input_size_sub_profile_features ,
                                                                  self.padding_sub_profile_features,
                                                                  self.kernel_size_sub_profile_features,
                                                                  self.stride_sub_profile_features)

        self.branch_hidden_rep_len = self.hidden_size_text + self.hidden_size_comments + self.hidden_size_users + \
                                     output_length_text*self.out_channels_text + \
                                     output_length_comments*self.out_channels_sub_features + \
                                     output_length_users*self.out_channels_sub_profile_features
        # number of labels - binary label 1/0 is event delta in branch discussion or not.
        self.num_labels = num_labels

        # linear output layers which projects back to tag space
        self.branch_hidden_to_linear_fc1 = nn.Linear(self.branch_hidden_rep_len, first_linear_reduction)
        self.fc2 = nn.Linear(first_linear_reduction, second_linear_reduction)
        self.fc3_to_label = nn.Linear(second_linear_reduction, self.num_labels)

    def forward(self, x):
        """
        this method performs all of the model logic and outputs the prediction
        :param x: [self.branch_comments_embedded_text_tensor[index], self.branch_comments_features_tensor[index],
        # self.branch_comments_user_profiles_tensor[index],
        # self.submission_data_dict[self.branch_submission_dict[index][0]] = [submission text, submission features,
        # submitter profile features], self.branch_submission_dict[index][1]] = [branch features]
        :return:
        """

        # divide input in a comprehensive way
        branch_comments_embedded_text = x[0]
        branch_comment_features_tensor = x[1]
        branch_comments_user_profiles_tensor = x[2]
        submission_embedded_text = x[3][0]
        submission_features = x[3][1]
        submitter_profile_features = x[3][2]
        branch_features_list = x[4]
        # TODO: FIX ACTUAL SIZES batch-> x[0] maybe better
        batch_size, seq_len, _ = branch_comments_embedded_text.size()
        # TODO: different parameter batch size than self.batch size?
        # initialize hidden between forward runs
        self.hidden_text = self.init_hidden(self.num_layers_text, batch_size, self.hidden_size_text)
        self.hidden_users = self.init_hidden(self.num_layers_users, batch_size, self.hidden_size_comments)
        self.hidden_comments = self.init_hidden(self.num_layers_comments, batch_size, self.hidden_size_users)

        # concatenate submission features and branch features as the convolution conv_sub_features input
        submission_branch_concat = tr.cat((submission_features, branch_features_list), 1)
        # TODO UNDERSTAND HOW TO GET BATCH SEQUENCES LENGTHS, can it be in the forward input?

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM, run the LSTM and unpack
        out_lstm_text = self.run_lstm_padded_packed(branch_comments_embedded_text, branch_comments_embedded_text_lengths
                                                    , 'text')

        out_lstm_comments = self.run_lstm_padded_packed(branch_comment_features_tensor,
                                                        branch_comment_features_tensor_lengths, 'comments')

        out_lstm_users = self.run_lstm_padded_packed(branch_comments_user_profiles_tensor,
                                                     branch_comments_user_profiles_tensor_lengths, 'users')

        # run through convolutions and activation functions
        out_sub_text = self.leaky_relu_text(self.conv_sub_text(submission_embedded_text))
        out_sub_features = self.leaky_relu_features(self.conv_sub_features(submission_branch_concat))
        out_sub_user = self.leaky_relu_user(self.conv_sub_user(submitter_profile_features))

        out_sub_text = out_sub_text.view(-1)
        out_sub_features = out_sub_features.view(-1)
        out_sub_user = out_sub_user.view(-1)

        # concatenate all LSTMs and convolutions outputs as input for final linear and softmax layer
        branch_hidden_rep = tr.cat((out_lstm_text, out_sub_text, out_lstm_comments, out_sub_features, out_lstm_users,
                                    out_sub_user), 1)

        # run through linear layers to get to label dimension
        output = self.branch_hidden_to_linear_fc1(branch_hidden_rep)
        output = self.fc2(output)
        prediction = self.fc3_to_label(output)

        # softmax
        prediction = F.log_softmax(prediction, dim=1)

        # reshape for mental sanity so we're back to (batch_size, seq_len, nb_tags)
        prediction = prediction.view(batch_size, seq_len, self.num_labels)

        return prediction

    def init_hidden(self, lstm_layers, batch_size, lstm_units):
        """
        initialize weights between forward runs of batches
        :param lstm_layers: number of lstm layers
        :param batch_size: number of samples in the batch
        :param lstm_units:
        :return:
        """
        # the weights are of the form (nb_lstm_layers, batch_size, nb_lstm_units)
        hidden_h = tr.randn(lstm_layers, batch_size, lstm_units)
        hidden_c = tr.randn(lstm_layers, batch_size, lstm_units)

        if self.hparams.on_gpu:
            hidden_h = hidden_h.cuda()
            hidden_c = hidden_c.cuda()

        hidden_h = hidden_h
        hidden_c = hidden_c

        return hidden_h, hidden_c

    def run_lstm_padded_packed(self, lstm_input, input_lengths, lstm_type):
        """
        this method uses the pack padded sequence functions to run the batch input of different sizes, padded with zeros
         in the LSTM
        :param lstm_input: the LSTM input
        :param input_lengths: the lengths of each sequence in the input batch
        :param lstm_type: which LSTM to run
        :return: the padded pack sequence output
        """

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        lstm_input = tr.nn.utils.rnn.pack_padded_sequence(lstm_input, input_lengths, batch_first=True)

        # run through LSTM_text
        if lstm_type == 'text':
            out_lstm, self.hidden_text = self.lstm_text(lstm_input, self.hidden_text)
        elif lstm_type == 'comments':
            out_lstm, self.hidden_comments = self.lstm_comments(lstm_input, self.hidden_comments)
        else:
            out_lstm, self.hidden_users = self.lstm_users(lstm_input, self.hidden_users)

        # undo the packing operation
        out_lstm, _ = tr.nn.utils.rnn.pad_packed_sequence(out_lstm, batch_first=True)

        # TODO: PROCESS LSTM OUTPUT DIMENSIONS:
        # # ---------------------
        # # 3. Project to tag space
        # # Dim transformation: (batch_size, seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_lstm_units)
        #
        # # this one is a bit tricky as well. First we need to reshape the data so it goes into the linear layer
        # TODO: it supposed to connect if the tensor indexes are not continuous - check if packed padded did something
        # TODO: that needs this kind of fix
        # out_lstm = out_lstm.contiguous()
        # out_lstm = out_lstm.view(-1, out_lstm.shape[2])
        # TODO: ensure lstm[-1] is correct
        return out_lstm[-1]

    def calc_conv1d_output_shape(self, batch_size, out_channels, input_size, padding, kernel_size, stride, dilation=1):
        """
        calculates output shape of the conv1d given its input and hyper parameters,
        calculation is based on formula from documentation: https://pytorch.org/docs/stable/nn.html#conv1d
        :param batch_size: size of batch in conv input
        :param out_channels: number of kernles
        :param input_size: vector size of inputs
        :param padding: zero padding added to both sides of the input
        :param kernel_size: size of kernel
        :param stride: kernel step
        :param dilation: gap between kernels
        :return:
        """

        length_out_seq = math.floor(((input_size+2*padding-dilation*(kernel_size-1)-1)/stride)+1)

        return batch_size, out_channels, length_out_seq

