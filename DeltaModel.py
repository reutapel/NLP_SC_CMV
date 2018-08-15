import torch as tr
import torch.nn as nn
from torch.nn import functional as F
import math
# TODO: add batch normalization


class DeltaModel(nn.Module):
    """
    A deep learning model class with the following logic of taking 3 different sectors of signals in data for predicting
    if the event "delta" occurred in the discussion "branch"
    """
    def __init__(self, init_lstm_text, init_lstm_comments, init_lstm_users, init_conv1d_text, init_conv1d_sub_features,
                 init_conv1d_sub_profile_features, input_size_text_sub, input_size_sub_features,
                 input_size_sub_profile_features, batch_size, num_labels, first_linear_reduction,
                 second_linear_reduction, fc1_dropout, fc2_dropout):
        """
        :param init_lstm_text: embedded text LSTM hyper parameters
        :param init_lstm_comments: comment features LSTM hyper parameters
        :param init_lstm_users: user features LSTM hyper parameters
        :param init_conv1d_text: submission embedded text conv hyper parameters
        :param init_conv1d_sub_features: submission features conv hyper parameters
        :param init_conv1d_sub_profile_features: submitter features conv hyper parameters
        :param input_size_text_sub: length of embedded text vector of submission
        :param input_size_sub_features: length of submission feature vector
        :param input_size_sub_profile_features: length of submitter feature vector
        :param batch_size: size of batch
        :param num_labels: number of labels
        :param first_linear_reduction: linear size from branch hidden rep output to this size
        :param second_linear_reduction: linear size from first_linear_size output to this size
        :param fc1_dropout: dropout probability before first linear layer
        :param fc2_dropout: dropout probability before second linear layer
        """
        super(DeltaModel, self).__init__()
        # check if gpu is available
        self.hparams.on_gpu = tr.cuda.is_available()

        self.init_lstm_text = init_lstm_text
        self.init_lstm_comments = init_lstm_comments
        self.init_lstm_users = init_lstm_users

        # initialize LSTM's hidden states
        self.hidden_text = self.init_hidden(self.init_lstm_text.num_layers, batch_size,
                                            self.init_lstm_text.hidden_size)
        self.hidden_comments = self.init_hidden(self.init_lstm_comments.num_layers, batch_size,
                                                self.init_lstm_comments.hidden_size)
        self.hidden_users = self.init_hidden(self.init_lstm_users.num_layers, batch_size,
                                             self.init_lstm_users.hidden_size)

        # define layers
        # static layers definition- aggregates the parameters for the derivatives
        # LSTM layers
        self.lstm_text = self.create_lstm_layer(self.init_lstm_text.input_size, self.init_lstm_text.hidden_size,
                                                self.init_lstm_text.num_layers,
                                                self.init_lstm_text.batch_first)

        self.lstm_comments = self.create_lstm_layer(self.init_lstm_comments.input_size,
                                                    self.init_lstm_comments.hidden_size,
                                                    self.init_lstm_comments.num_layers,
                                                    self.init_lstm_comments.batch_first)

        self.lstm_users = self.create_lstm_layer(self.init_lstm_users.input_size, self.init_lstm_users.hidden_size,
                                                 self.init_lstm_users.num_layers, self.init_lstm_users.batch_first)

        # convolution layers
        self.conv_sub_text = self.create_conv1d_layer(init_conv1d_text.in_channels, init_conv1d_text.out_channels,
                                                      init_conv1d_text.kernel_size, init_conv1d_text.stride,
                                                      init_conv1d_text.padding)

        self.conv_sub_features = self.create_conv1d_layer(init_conv1d_sub_features.in_channels,
                                                          init_conv1d_sub_features.out_channels,
                                                          init_conv1d_sub_features.kernel_size,
                                                          init_conv1d_sub_features.stride,
                                                          init_conv1d_sub_features.padding)

        self.conv_sub_user = self.create_conv1d_layer(init_conv1d_sub_profile_features.in_channels,
                                                      init_conv1d_sub_profile_features.out_channels,
                                                      init_conv1d_sub_profile_features.kernel_size,
                                                      init_conv1d_sub_profile_features.stride,
                                                      init_conv1d_sub_profile_features.padding)
        # activation layers
        self.leaky_relu_text = nn.LeakyReLU(init_conv1d_text.leakyRelu)
        self.leaky_relu_features = nn.LeakyReLU(init_conv1d_sub_features.leakyRelu)
        self.leaky_relu_user = nn.LeakyReLU(init_conv1d_sub_profile_features.leakyRelu)

        # calculate the output length of the convolutions for the definition of the linear layers dimensions
        _, _, output_length_text = self.calc_conv1d_output_shape(self.batch_size, init_conv1d_text.out_channels,
                                                                 input_size_text_sub, init_conv1d_text.padding,
                                                                 init_conv1d_text.kernel_size, init_conv1d_text.stride)
        _, _, output_length_comments = self.calc_conv1d_output_shape(self.batch_size,
                                                                     init_conv1d_sub_features.out_channels,
                                                                     input_size_sub_features,
                                                                     init_conv1d_sub_features.padding,
                                                                     init_conv1d_sub_features.kernel_size,
                                                                     init_conv1d_sub_features.stride)
        _, _, output_length_users = self.calc_conv1d_output_shape(self.batch_size,
                                                                  init_conv1d_sub_profile_features.out_channels,
                                                                  input_size_sub_profile_features,
                                                                  init_conv1d_sub_profile_features.padding,
                                                                  init_conv1d_sub_profile_features.kernel_size,
                                                                  init_conv1d_sub_profile_features.stride)

        self.branch_hidden_rep_len = self.init_lstm_text.hidden_size + self.init_lstm_comments.hidden_size + \
                                     self.init_lstm_users.hidden_size + \
                                     output_length_text*init_conv1d_text.out_channels + \
                                     output_length_comments*init_conv1d_sub_features.out_channels + \
                                     output_length_users*init_conv1d_sub_profile_features.out_channels

        # number of labels - binary label 1/0 is event delta in branch discussion or not.
        self.num_labels = num_labels

        # dropout regularization
        self.droput_prior_fc1 = nn.Dropout(fc1_dropout)
        self.droput_prior_fc2 = nn.Dropout(fc2_dropout)

        # linear output layers which projects back to tag space
        self.branch_hidden_to_linear_fc1 = nn.Linear(self.branch_hidden_rep_len, first_linear_reduction)
        self.fc2 = nn.Linear(first_linear_reduction, second_linear_reduction)
        self.fc3_to_label = nn.Linear(second_linear_reduction, self.num_labels)

    def create_lstm_layer(self, input_size, hidden_size, num_layers, batch_first):
        """
        initialize LSTM layer with given argumnets
        input_size: embedding dimension or feature vector dimension
        hidden_size: The number of features in the hidden state `h`
        num_layers: number of lstm layers
        batch_first: True if batch dimension is first
        :return: lstm layer
        """
        lstm_layer = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first)

        return lstm_layer

    def create_conv1d_layer(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        initialize conv1d layer with given argumnets
        in_channels: number of channels in the input
        out_channels: number of channels produced by the convolution (number of filters / kernels)
        kernel_size: sliding window dimension
        stride: step size of kernel
        padding: zero padding added to both sides of the input
        :return: conv1d layer
        """
        conv1d_layer = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

        return conv1d_layer

    def forward(self, x):
        """
        this method performs all of the model logic and outputs the prediction
        :param x: [self.branch_comments_embedded_text_tensor[index], self.branch_comments_features_tensor[index],
                   self.branch_comments_user_profiles_tensor[index],
                   self.submission_data_dict[self.branch_submission_dict[index][0]] = [submission text, submission
                   features, submitter profile features], self.branch_submission_dict[index][1] = [branch features],
                   self.branches_lengths[index]]
        :return: prediction on x
        """
        # TODO: check if another indexation is needed cuz of tensor wrap in get item
        # TODO: check how batch is handled
        # divide input in a comprehensive way
        branch_comments_embedded_text = x[:, 0]
        branch_comment_features_tensor = x[1]
        branch_comments_user_profiles_tensor = x[2]
        submission_embedded_text = x[3][0]
        submission_features = x[3][1]
        submitter_profile_features = x[3][2]
        branch_features_list = x[4]
        branches_lengths = x[5]
        # TODO: FIX ACTUAL SIZES batch-> x[0] maybe better
        batch_size, seq_len, _ = branch_comments_embedded_text.size()

        # initialize hidden between forward runs
        self.hidden_text = self.init_hidden(self.init_lstm_text.num_layers, batch_size,
                                            self.init_lstm_text.hidden_size)
        self.hidden_comments = self.init_hidden(self.init_lstm_comments.num_layers, batch_size,
                                                self.init_lstm_comments.hidden_size)
        self.hidden_users = self.init_hidden(self.init_lstm_users.num_layers, batch_size,
                                             self.init_lstm_users.hidden_size)

        # concatenate submission features and branch features as the convolution conv_sub_features input
        submission_branch_concat = tr.cat((submission_features, branch_features_list), 1)

        # TODO: sort sequences by length and get length by input that is a shuffled batch

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM, run the LSTM and unpack
        out_lstm_text = self.run_lstm_padded_packed(branch_comments_embedded_text, branches_lengths
                                                    , 'text')

        out_lstm_comments = self.run_lstm_padded_packed(branch_comment_features_tensor,
                                                        branches_lengths, 'comments')

        out_lstm_users = self.run_lstm_padded_packed(branch_comments_user_profiles_tensor,
                                                     branches_lengths, 'users')

        # run through convolutions and activation functions
        out_sub_text = self.leaky_relu_text(self.conv_sub_text(submission_embedded_text))
        out_sub_features = self.leaky_relu_features(self.conv_sub_features(submission_branch_concat))
        out_sub_user = self.leaky_relu_user(self.conv_sub_user(submitter_profile_features))

        #TODO: chack batch flatten is correct in dimensions
        out_sub_text = out_sub_text.view(out_sub_text.size(0), -1)
        out_sub_features = out_sub_features.view(out_sub_text.size(0), -1)
        out_sub_user = out_sub_user.view(out_sub_text.size(0), -1)

        # concatenate all LSTMs and convolutions outputs as input for final linear and softmax layer
        branch_hidden_rep = tr.cat((out_lstm_text, out_sub_text, out_lstm_comments, out_sub_features, out_lstm_users,
                                    out_sub_user), 1)

        # run through linear layers to get to label dimension
        output = self.droput_prior_fc1(branch_hidden_rep)
        output = self.branch_hidden_to_linear_fc1(output)
        output = self.droput_prior_fc2(output)
        output = self.fc2(output)
        prediction = self.fc3_to_label(output)

        # softmax
        prediction = F.log_softmax(prediction, dim=1)

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
        # TODO: check if batch first = true means that even here it is first
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

        # TODO: it supposed to connect if the tensor indexes are not continuous - check if packed padded did something
        # TODO: that needs this kind of fix
        # out_lstm = out_lstm.contiguous()
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
        # TODO: check that L_in is really input size
        length_out_seq = math.floor(((input_size+2*padding-dilation*(kernel_size-1)-1)/stride)+1)

        return batch_size, out_channels, length_out_seq

