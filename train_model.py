import torch as tr
import torch.nn as nn
from model_utils import CustomDataset
from torch.utils import data as dt
from DeltaModel import DeltaModel
from model_utils import InitLstm
from model_utils import InitConv1d
import joblib
import pandas as pd
from tqdm import tqdm
from time import gmtime, strftime
from sklearn import metrics


# TODO: F.mse_loss(size_average, reduce) : parameters that affect if we get average values per batch : sum or average

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

        # calculate number of trainable paramaters
        print(sum(p.numel() for p in self.model.parameters() if p.requires_grad), " trainable parameters in model")


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

        for epoch in tqdm(range(self.num_epochs), desc='epoch'):

            # TODO: if want to send to test in each epoch need to do again self.model.train()
            for i, (data_points, labels) in tqdm(enumerate(self.train_loader), desc='batch'):

                # forward
                outputs, sorted_idx, batch_size = self.model(data_points)
                # TODO: understand impact of packed padded to loss, like function loss in model.py

                # sort labels
                labels = labels[sorted_idx].reshape(batch_size, -1)

                # calculate loss
                print("calc loss")
                loss = self.criterion(outputs.reshape(batch_size, -1), labels.float())

                # initialize gradient so only current batch will be summed and then backward
                self.optimizer.zero_grad()

                # calculate gradients
                loss.backward()

                # update parameters : tensor - learning_rate*gradient
                self.optimizer.step()

                if (i+1) % 100 == 0:
                    print('Epoch: [%d%d], Step: [%d%d], Loss: %.4f' % (epoch+1, self.num_epochs, i+1,
                                                                       len(self.train_dataset)//self.batch_size,
                                                                       loss.data[0]))

                # TODO: add graph of train/test auc, precision, recall per epoch
                # TODO: call test after each epoch

    def test(self):
        """
        test model on test data calculate accuracy
        :return:
        """
        print("start evaluation on test", strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()))
        # doesn't save history for backwards, turns off dropouts
        self.model.eval()

        correct = 0
        total = 0
        test_labels = tr.Tensor()
        test_predictions = tr.Tensor()
        for data_points, labels in self.test_loader:

            outputs, sorted_idx, batch_size = self.model(data_points)
            outputs = outputs.reshape(batch_size, -1).float()

            labels = labels[sorted_idx].reshape(batch_size, -1).float()

            predicted = (outputs > 0.5).float() * 1
            total += labels.size(0)
            correct += (predicted == labels).sum()
            test_labels = tr.cat((test_labels, labels))
            test_predictions = tr.cat((test_predictions, outputs))

        # calculate measurements on test data
        print('Accuracy of the model on the test: %d %%' % (100 * correct / total))
        fpr, tpr, thresholds = metrics.roc_curve(test_labels.detach().numpy(), test_predictions.detach().numpy(),
                                                 pos_label=1)
        print("AUC is: ", metrics.auc(fpr, tpr))

        # save the model
        tr.save(self.model.state_dict(), 'model.pkl')


def main():

    debug = 1
    if not debug:
        # load train data
        branch_comments_embedded_text_df_train = joblib.load("branch_comments_embedded_text_df_train")
        branch_comments_features_df_train = joblib.load("branch_comments_features_df_train")
        branch_comments_user_profiles_df_train = joblib.load("branch_comments_user_profiles_df_train")
        branch_submission_dict_train = joblib.load("branch_submission_dict_train")
        submission_data_dict_train = joblib.load("submission_data_dict_train")
        branch_deltas_data_dict_train = joblib.load("branch_deltas_data_dict_train")
        branches_lengths_list_train = joblib.load("branches_lengths_list_train")

        # load test data
        branch_comments_embedded_text_df_test = joblib.load("branch_comments_embedded_text_df_test")
        branch_comments_features_df_test = joblib.load("branch_comments_features_df_test")
        branch_comments_user_profiles_df_test = joblib.load("branch_comments_user_profiles_df_test")
        branch_submission_dict_test = joblib.load("branch_submission_dict_test")
        submission_data_dict_test = joblib.load("submission_data_dict_test")
        branch_deltas_data_dict_test = joblib.load("branch_deltas_data_dict_test")
        branches_lengths_list_test = joblib.load("branches_lengths_list_test")

    else:
        # DEBUG
        # load train data
        text = {'col1': [[1., 2.], [3., 4.], [5., 6.]], 'col2': [[7., 8.], [9., 1.], [1., 5.]], 'col3': [[1., 4.], [1., 6.], [0., 0.]],
                'col4': [[9., 2.], [0., 0.], [0., 0.]]}
        comment_features = {'col1': [[10., 20., 34.], [30., 40., 54.], [50., 60., 70.]], 'col2': [[70., 80., 82.], [90., 10., 43.],
                                                                                         [11., 12., 65.]],
                            'col3': [[13., 14., 76.], [15., 16., 17.], [0., 0., 0.]],
                            'col4': [[19., 20., 65.], [0., 0., 0.], [0., 0., 0.]]}
        users_profiles = {'col1': [[111., 222.], [333., 444.], [555., 666.]], 'col2': [[777., 888.], [999., 100.], [111., 122.]],
                          'col3': [[133., 144.], [155., 166.], [0., 0.]],
                          'col4': [[190., 200.], [0., 0.], [0., 0.]]}
        branch_comments_embedded_text_df_train = pd.DataFrame(data=text)
        branch_comments_features_df_train = pd.DataFrame(data=comment_features)
        branch_comments_user_profiles_df_train = pd.DataFrame(data=users_profiles)
        branch_submission_dict_train = {0: ['6ax', [-4., 4., -44., 44.]], 1: ['6ax', [-5., 5., -55., 55.]], 2: ['3f9', [-2., 2., -22., 22.]]}
        submission_data_dict_train = {'6ax': [[12., 21.], [6., 9.], [1., 11., 111.]], '3f9': [[13., 31.], [66., 99.], [1., 11., 111.]]}
        branch_deltas_data_dict_train = {0: [1., 2, [1, 3]], 1: [0., 0, []], 2: [1., 1, [1]]}
        branches_lengths_list_train = [4, 3, 2]

        # load test data
        branch_comments_embedded_text_df_test = pd.DataFrame(data=text)
        branch_comments_features_df_test = pd.DataFrame(data=comment_features)
        branch_comments_user_profiles_df_test = pd.DataFrame(data=users_profiles)
        branch_submission_dict_test = {0: ['6ax', [-4, 4, -44, 44]], 1: ['6ax', [-5, 5, -55, 55]], 2: ['3f9', [-2, 2, -22, 22]]}
        submission_data_dict_test = {'6ax': [[12, 21], [6, 9], [1, 11, 111]], '3f9': [[13, 31], [66, 99], [1, 11, 111]]}
        branch_deltas_data_dict_test = {0: [1, 2, ], 1: [0, 0, 0], 2: [1, 1, ]}
        branches_lengths_list_test = [4, 3, 2]

    train_data = branch_comments_embedded_text_df_train, branch_comments_features_df_train, \
                 branch_comments_user_profiles_df_train, branch_submission_dict_train, submission_data_dict_train, \
                 branch_deltas_data_dict_train, branches_lengths_list_train

    test_data = branch_comments_embedded_text_df_test, branch_comments_features_df_test, \
                branch_comments_user_profiles_df_test, branch_submission_dict_test, submission_data_dict_test, \
                branch_deltas_data_dict_test, branches_lengths_list_test

    # define hyper parameters of learning phase
    criterion = nn.BCEWithLogitsLoss()
    learning_rate = 0.01
    batch_size = 2
    num_epochs = 3
    num_labels = 2
    fc1 = 32
    fc2 = 16
    fc1_dropout = 0.2
    fc2_dropout = 0.5

    # define LSTM layers hyperparameters
    init_lstm_text = InitLstm(input_size=2, hidden_size=3, num_layers=3, batch_first=True)
    init_lstm_comments = InitLstm(input_size=3, hidden_size=3, num_layers=5, batch_first=True)
    init_lstm_users = InitLstm(input_size=2, hidden_size=3, num_layers=4, batch_first=True)

    # define conv layers hyperparameters
    init_conv_text = InitConv1d(in_channels=1, out_channels=4, kernel_size=1, stride=1, padding=0,
                                leaky_relu_alpha=0.001)
    init_conv_sub_features = InitConv1d(in_channels=1, out_channels=4, kernel_size=1, stride=1, padding=0,
                                        leaky_relu_alpha=0.001)
    init_conv_sub_profile_features = InitConv1d(in_channels=1, out_channels=4, kernel_size=1, stride=1, padding=0,
                                                leaky_relu_alpha=0.001)

    input_size_text_sub = 2
    input_size_sub_features = 6  # submission features + branch features
    input_size_sub_profile_features = 3

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

