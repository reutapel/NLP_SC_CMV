import torch as tr
import torch.nn as nn
from torch.autograd import Variable
from model_utils import CustomDataset
from torch.utils import data as dt
from torch.nn import functional as F


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
                 stride_sub_profile_features, padding_sub_profile_features, num_labels):

        super(DeltaModel, self).__init__()

        self.hparams.on_gpu = tr.cuda.is_available()

        # define lstm's parameters
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

        # define layers
        # static layers definition- aggregates the parameters for the derivatives

        # lstm layers
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
        # out_channels: number of channels produced by the convolution
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

        self.branch_hidden_rep_len = 6 #TODO: CALCULATE THE CONCATENATED branch_hidden_rep VECTOR SIZE
        self.num_labels = num_labels

        # output layer which projects back to tag space
        self.branch_hidden_to_tag = nn.Linear(self.branch_hidden_rep_len, self.num_labels)

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

        batch_size, seq_len, _ = branch_comments_embedded_text.size()

        self.hidden_text = self.init_hidden(self.num_layers_text, batch_size, ) # TODO: fill last argument num of lstm units
        self.hidden_users = self.init_hidden(self.num_layers_users, batch_size, )# TODO: fill last argument num of lstm units
        self.hidden_comments = self.init_hidden(self.num_layers_comments, batch_size, )# TODO: fill last argument num of lstm units

        # concatenate submission features and branch features as the convolution conv_sub_features input
        submission_branch_concat = tr.cat((submission_features, branch_features_list), 1)
        # TODO UNDERSTAND HOW TO GET BATCH SEQUENCES LENGTHS, can it be in the forward input?
        # TODO: initialize hidden of all of LSTMS
        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        out_lstm_text = self.run_lstm_padded_packed(branch_comments_embedded_text, branch_comments_embedded_text_lengths
                                                    , 'text')

        out_lstm_comments = self.run_lstm_padded_packed(branch_comment_features_tensor,
                                                        branch_comment_features_tensor_lengths, 'comments')

        out_lstm_users = self.run_lstm_padded_packed(branch_comments_user_profiles_tensor,
                                                     branch_comments_user_profiles_tensor_lengths, 'users')

        # run through convolutions and activation functions
        out_sub_text = nn.LeakyReLU(self.conv_sub_text(submission_embedded_text))
        out_sub_features = nn.LeakyReLU(self.conv_sub_features(submission_branch_concat))
        out_sub_user = nn.LeakyReLU(self.conv_sub_user(submitter_profile_features))

        # concatenate all LSTMs and convolutions outputs as input for final linear and softmax layer
        branch_hidden_rep = tr.cat((out_lstm_text, out_sub_text, out_lstm_comments, out_sub_features, out_lstm_users,
                                    out_sub_user), 1)

        # run through linear layer to get to label dimension
        prediction = self.branch_hidden_to_tag(branch_hidden_rep)

        # softmax
        prediction = F.log_softmax(prediction, dim=1)

        # I like to reshape for mental sanity so we're back to (batch_size, seq_len, nb_tags)
        prediction = prediction.view(batch_size, seq_len, self.num_labels)

        return prediction

    def init_hidden(self, lstm_layers, batch_size, lstm_units):
        # the weights are of the form (nb_lstm_layers, batch_size, nb_lstm_units)
        hidden_a = tr.randn(lstm_layers, batch_size, lstm_units)
        hidden_b = tr.randn(lstm_layers, batch_size, lstm_units)

        if self.hparams.on_gpu:
            hidden_a = hidden_a.cuda()
            hidden_b = hidden_b.cuda()

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)

    def run_lstm_padded_packed(self, input, input_lengths, lstm_type):

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        input = tr.nn.utils.rnn.pack_padded_sequence(input, input_lengths, batch_first=True)
        # run through LSTM_text
        if lstm_type == 'text':
            out_lstm, self.hidden_text = self.lstm_text(input, self.hidden_text)
        elif lstm_type == 'comments':
            out_lstm, self.hidden_comments = self.lstm_comments(input, self.hidden_comments)
        else:
            out_lstm, self.hidden_users = self.lstm_users(input, self.hidden_users)

        # undo the packing operation
        out_lstm, _ = tr.nn.utils.rnn.pad_packed_sequence(out_lstm, batch_first=True)

        # TODO: PROCESS LSTM OUTPUT DIMENSIONS:
        # # ---------------------
        # # 3. Project to tag space
        # # Dim transformation: (batch_size, seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_lstm_units)
        #
        # # this one is a bit tricky as well. First we need to reshape the data so it goes into the linear layer
        # out_lstm = out_lstm.contiguous()
        # out_lstm = out_lstm.view(-1, out_lstm.shape[2])

        return out_lstm


# TODO: ensure zero weights initializing between batches


learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
model=DeltaModel()
# on the model parameters will be performed the update
optimizer = tr.optim.SGD(model.parameters(), lr=learning_rate)
batch_size = 128
num_epochs = 100

train_dataset = CustomDataset(branch_comments_embedded_text_df, branch_comments_features_df,
                              branch_comments_user_profiles_df, branch_submission_dict, submission_data_dict,
                              branch_deltas_data_dict)

test_dataset = CustomDataset(branch_comments_embedded_text_df, branch_comments_features_df,
                              branch_comments_user_profiles_df, branch_submission_dict, submission_data_dict,
                              branch_deltas_data_dict)


train_loader = dt.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = dt.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)



# training

device = tr.device("cuda" if tr.cuda.is_available() else "cpu")

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # -1 so last batch will fit the size
        images = Variable(images.view(-1, 28*28))
        labels = Variable(labels)
        # forward + backward + optimize
        # initialize gradient so only current batch will be summed and then backward
        optimizer.zero_grad()
        # forward
        outputs = model(images)
        # TODO: understand impact of packed padded to loss, like function loss in model.py
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i+1) % 100 ==0:
            print('Epoch: [%d%d], Step: [%d%d], Loss: %.4f' % (epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))


# testing
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.view(-1, 28*28))
    outputs = model(images)
    _, predicted = tr.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
    break
print('Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))

# save the model
tr.save(model.state_dict(), 'model.pkl')

# TODO: add batch normalization
# TODO: add metric learning for a richer representation of the label
# TODO: F.mse_loss(size_average, reduce) : parameters that affect if we get average values per batch : sum or average