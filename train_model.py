import torch as tr
import torch.nn as nn
from torch.autograd import Variable
from model_utils import CustomDataset
from torch.utils import data as dt
from DeltaModel import DeltaModel
from model_utils import InitLstm
from model_utils import InitConv1d
import joblib

# TODO: F.mse_loss(size_average, reduce) : parameters that affect if we get average values per batch : sum or average
# TODO SGD dynamic to something else?
class TrainModel:
    """
    class builds the data sets, data loaders, model, trains and tests the model.
    """
    def __init__(self, train_data, test_data, learning_rate, criterion, batch_size, num_epochs, num_labels, fc1, fc2,
                 init_lstm_text, init_lstm_comments, init_lstm_users, init_conv_text, init_conv_sub_features,
                 init_conv_sub_profile_features, input_size_text_sub, input_size_sub_features,
                 input_size_sub_profile_features, fc1_droput, fc2_dropout):
        """

        :param train_data: all data elements of train
        :param test_data: all data elements of test
        :param learning_rate: learning rate pace
        :param criterion: which criterion to calculate loss by
        :param batch_size: size of batch
        :param num_epochs: number of epochs to train
        :param num_labels: number of labels in data
        :param fc1: first linear reduction size from concatenated hidden size to
        :param fc2: second reduction , before reduction to label dimension of 2
        :param init_lstm_text: hyper parameters initializer class instance
        :param init_lstm_comments: hyper parameters initializer class instance
        :param init_lstm_users: hyper parameters initializer class instance
        :param init_conv_text: hyper parameters initializer class instance
        :param init_conv_sub_features: hyper parameters initializer class instance
        :param init_conv_sub_profile_features: hyper parameters initializer class instance
        :param input_size_text_sub: size of embedded text vector
        :param input_size_sub_features: size of submission feature vector
        :param input_size_sub_profile_features: size of submitter feature vector
        :param fc1_droput: probability for first dropout
        :param fc2_dropout: probability for second dropout
        """

        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.criterion = criterion

        # craete model
        self.model = DeltaModel(init_lstm_text, init_lstm_comments, init_lstm_users, init_conv_text,
                                init_conv_sub_features, init_conv_sub_profile_features, input_size_text_sub,
                                input_size_sub_features, input_size_sub_profile_features, self.batch_size, num_labels,
                                fc1, fc2, fc1_droput, fc2_dropout)

        # on the model.parameters will be performed the update by stochastic gradient descent
        self.optimizer = tr.optim.SGD(self.model.parameters(), lr=learning_rate)

        # create datesets
        self.train_dataset = self.create_dataset(train_data)
        self.test_dataset = self.create_dataset(test_data)

        # create data loaders
        self.train_loader = self.create_data_loader(self.train_dataset, self.batch_size)
        self.test_loader =  self.create_data_loader(self.test_dataset, self.batch_size)


    def create_dataset(self, data):
        """

        :param data: all the data structures needed for the class
        :return: CustomDataset object
        """

        return CustomDataset(*data)


    def create_data_loader(self, dataset, batch_size):
        """

        :param dataset: dataset train or test for data loader to run over
        :param batch_size: size of batch
        :return: data loader object
        """

        return dt.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


    def train(self):
        """
        train the model on train data by batches iterate all epochs
        :return:
        """

        self.model.train()

        #device = tr.device("cuda" if tr.cuda.is_available() else "cpu")

        for epoch in range(self.num_epochs):
            # TODO: if want to send to test in each epoch need to do again self.model.train()
            for i, (data_points, labels) in enumerate(self.train_loader):

                # initialize gradient so only current batch will be summed and then backward
                self.optimizer.zero_grad()

                # forward
                outputs = self.model(data_points)
                # TODO: understand impact of packed padded to loss, like function loss in model.py
                # calculate loss
                loss = self.criterion(outputs, labels)
                # calculate gradients
                loss.backward()
                # update parameters : tensor - learning_rate*gradient
                self.optimizer.step()

                if (i+1) % 100 ==0:
                    print('Epoch: [%d%d], Step: [%d%d], Loss: %.4f' % (epoch+1, self.num_epochs, i+1,
                                                                       len(self.train_dataset)//self.batch_size,
                                                                       loss.data[0]))

    def test(self):
        """
        test model on test data calculate accuracy
        :return:
        """
        # doesn't save history for backwards, turns off dropouts
        self.model.eval()

        correct = 0
        total = 0
        for data_points, labels in self.test_loader:

            outputs = self.model(data_points)
            # TODO: take sigmoid output and round , correct following row:
            _, predicted = tr.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print('Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))
        # TODO: add AUC
        # save the model
        tr.save(self.model.state_dict(), 'model.pkl')


def main():

    # load train data
    branch_comments_embedded_text_df_train = joblib.load("branch_comments_embedded_text_df_train")
    branch_comments_features_df_train = joblib.load("branch_comments_features_df_train")
    branch_comments_user_profiles_df_train = joblib.load("branch_comments_user_profiles_df_train")
    branch_submission_dict_train = joblib.load("branch_submission_dict_train")
    submission_data_dict_train = joblib.load("submission_data_dict_train")
    branch_deltas_data_dict_train = joblib.load("branch_deltas_data_dict_train")

    # load test data
    branch_comments_embedded_text_df_test = joblib.load("branch_comments_embedded_text_df_test")
    branch_comments_features_df_test = joblib.load("branch_comments_features_df_test")
    branch_comments_user_profiles_df_test = joblib.load("branch_comments_user_profiles_df_test")
    branch_submission_dict_test = joblib.load("branch_submission_dict_test")
    submission_data_dict_test = joblib.load("submission_data_dict_test")
    branch_deltas_data_dict_test = joblib.load("branch_deltas_data_dict_test")

    train_data = branch_comments_embedded_text_df_train, branch_comments_features_df_train, \
                 branch_comments_user_profiles_df_train, branch_submission_dict_train, submission_data_dict_train, \
                 branch_deltas_data_dict_train

    test_data = branch_comments_embedded_text_df_test, branch_comments_features_df_test, \
                branch_comments_user_profiles_df_test, branch_submission_dict_test,submission_data_dict_test, \
                branch_deltas_data_dict_test

    # define hyper parameters of learning phase
    # TODO: maybe it needs one value as input so replace softmax with sigmoid, and finish linear in dimension of 1
    # TODO: CE with LOGIT
    criterion = nn.BCELoss()
    learning_rate = 0.01
    batch_size = 128
    num_epochs = 100
    num_labels = 2
    fc1 = 128
    fc2 = 32
    fc1_dropout = 0.2
    fc2_dropout = 0.5

    # define LSTM layers hyperparameters
    init_lstm_text = InitLstm( input_size=50, hidden_size=30, num_layers=5, batch_first=True)
    init_lstm_comments = InitLstm( input_size=50, hidden_size=30, num_layers=5, batch_first=True)
    init_lstm_users = InitLstm( input_size=50, hidden_size=30, num_layers=5, batch_first=True)

    # define conv layers hyperparameters
    init_conv_text = InitConv1d(in_channels=1, out_channels=6, kernel_size=2, stride=2, padding=1,
                                leaky_relu_alpha=0.001)
    init_conv_sub_features = InitConv1d(in_channels=1, out_channels=6, kernel_size=2, stride=2, padding=1,
                                        leaky_relu_alpha=0.001)
    init_conv_sub_profile_features = InitConv1d(in_channels=1, out_channels=6, kernel_size=2, stride=2, padding=1,
                                                leaky_relu_alpha=0.001)

    input_size_text_sub = 50
    input_size_sub_features = 10
    input_size_sub_profile_features = 8

    # create training instance
    train_model = TrainModel(train_data, test_data, learning_rate, criterion, batch_size, num_epochs, num_labels, fc1,
                             fc2,init_lstm_text, init_lstm_comments, init_lstm_users, init_conv_text,
                             init_conv_sub_features, init_conv_sub_profile_features, input_size_text_sub,
                             input_size_sub_features, input_size_sub_profile_features, fc1_dropout, fc2_dropout)

    # train and test model
    train_model.train()
    train_model.test()

if __name__ == '__main__':
    main()

