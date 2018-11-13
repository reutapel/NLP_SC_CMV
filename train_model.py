import torch as tr
import torch.nn as nn
from model_utils import CustomDataset
from torch.utils import data as dt
from DeltaModel import DeltaModel
from model_utils import InitLstm
from model_utils import InitConv1d
import pandas as pd
from tqdm import tqdm
from time import gmtime, strftime
from sklearn import metrics
import joblib
import pickle
import matplotlib.pyplot as plt
from pylab import savefig
from matplotlib.ticker import MaxNLocator
import sys
import os
import ast
import numbers

# old_stdout = sys.stdout
# log_file = open("train_model.log", "w")
# sys.stdout = log_file


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

        # create model
        self.model = DeltaModel(init_lstm_text, init_lstm_comments, init_lstm_users, init_conv_text,
                                init_conv_sub_features, init_conv_sub_profile_features, input_size_text_sub,
                                input_size_sub_features, input_size_sub_profile_features, self.batch_size, num_labels,
                                fc1, fc2, fc1_droput, fc2_dropout)

        # on the model.parameters will be performed the update by stochastic gradient descent
        self.optimizer = tr.optim.SGD(self.model.parameters(), lr=learning_rate)

        # create datasets
        self.train_dataset = self.create_dataset(train_data)
        self.test_dataset = self.create_dataset(test_data)

        # create data loaders
        self.train_loader = self.create_data_loader(self.train_dataset, self.batch_size)
        self.test_loader = self.create_data_loader(self.test_dataset, self.batch_size)
        self.train_loss_list = list()
        self.test_loss_list = list()
        self.measurements_dict = dict()
        self.measurements_dict["train"] = dict()
        self.measurements_dict["test"] = dict()

        # calculate number of trainable parameters
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

        # device = tr.device("cuda" if tr.cuda.is_available() else "cpu")

        for epoch in tqdm(range(self.num_epochs), desc='epoch'):

            print("start epoch number ", epoch)

            correct = 0
            total = 0
            train_labels = tr.Tensor()
            train_predictions = tr.Tensor()
            first_batch = True
            for i, (data_points, labels) in tqdm(enumerate(self.train_loader), desc='batch'):

                # forward
                outputs, sorted_idx, batch_size = self.model(data_points)
                # TODO: understand impact of packed padded to loss, like function loss in model.py

                outputs = outputs.reshape(batch_size, -1).float()

                # sort labels
                labels = labels[sorted_idx].reshape(batch_size, -1).float()

                # calculate for measurements
                predicted = (outputs > 0.5).float() * 1
                total += labels.size(0)
                correct += (predicted == labels).sum()
                train_labels = tr.cat((train_labels, labels))
                train_predictions = tr.cat((train_predictions, predicted))

                # calculate loss
                # print("calc loss")
                loss = self.criterion(outputs, labels)
                if first_batch:
                    first_batch = False
                    self.train_loss_list.append(loss.data[0].item())
                else:
                    self.train_loss_list[epoch] += loss.data[0].item()


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

            # calculate measurements on train data
            self.calc_measurements(correct, total, train_labels, train_predictions, epoch,  "train")
            self.test(epoch)
            self.model.train()

        # save the model
        tr.save(self.model.state_dict(), 'model.pkl')

    def test(self, epoch):
        """
        test model on test data
        :return:
        """
        print("start evaluation on test", strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()))

        # doesn't save history for backwards, turns off dropouts
        self.model.eval()
        first_batch = True

        correct = 0
        total = 0
        test_labels = tr.Tensor()
        test_predictions = tr.Tensor()
        for data_points, labels in self.test_loader:

            outputs, sorted_idx, batch_size = self.model(data_points)
            outputs = outputs.reshape(batch_size, -1).float()

            labels = labels[sorted_idx].reshape(batch_size, -1).float()

            # calculate loss
            loss = self.criterion(outputs, labels)
            if first_batch:
                first_batch = False
                self.test_loss_list.append(loss.data[0].item())
            else:
                self.test_loss_list[epoch] += loss.data[0].item()

            predicted = (outputs > 0.5).float() * 1
            total += labels.size(0)
            correct += (predicted == labels).sum()
            test_labels = tr.cat((test_labels, labels))
            # TODO: think why it was with outputs: test_predictions = tr.cat((test_predictions, outputs))
            test_predictions = tr.cat((test_predictions, predicted))

        # calculate measurements on test data
        self.calc_measurements(correct, total, test_labels, test_predictions, epoch, "test")

    def calc_measurements(self, correct, total, labels, pred, epoch, dataset):

        labels = labels.detach().numpy()
        pred = pred.detach().numpy()
        print("calculate measurements on ", dataset)
        accuracy = correct / total
        print('Accuracy: %d %%' % (100 * accuracy))
        fpr, tpr, thresholds = metrics.roc_curve(labels, pred,
                                                 pos_label=1)
        auc = metrics.auc(fpr, tpr)
        print("AUC: ", auc)

        precision = metrics.precision_score(labels, pred)
        print("precision: ", precision)

        recall = metrics.recall_score(labels, pred)
        print("recall: ", recall)

        self.measurements_dict[dataset][epoch] = [accuracy, auc, precision, recall]

    def plot_loss(self, epoch_count, train_loss, test_loss):

        # Visualize loss history
        plt.plot(list(range(epoch_count)), train_loss, 'g--')
        plt.plot(list(range(epoch_count)), test_loss, 'b-')
        plt.legend(['Training Loss', 'Test Loss'])

        # set ticks as int
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        # handle axis labels
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12, rotation='horizontal', verticalalignment='bottom')
        plt.gca().yaxis.set_label_coords(0, 1.01)
        # plt.gca().xaxis.set_label_coords(0.5, -0.02)

        for i in ['top', 'right']:
            plt.gca().spines[i].set_visible(False)
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.05), ncol=1, frameon=True)

        savefig('loss_graph.png')


def replace_0_with_list(df, len_list_in_cell):
    for i, row in enumerate(df.values):
        for j, col in enumerate(row):
            if isinstance(col, numbers.Number):
                df.loc[i, j] = [0] * len_list_in_cell
    return df


def main():

    debug = 0
    base_dir = os.path.abspath(os.curdir)
    features_dir = os.path.join(base_dir, "features", "small_data_features")
    if not debug:
        # load train data
        branch_comments_embedded_text_df_train = joblib.load(os.path.join(features_dir, "branch_comments_embedded_text_df_train.pkl"))
        branch_comments_features_df_train = joblib.load(os.path.join(features_dir, "branch_comments_features_df_train.pkl"))
        branch_comments_user_profiles_df_train = joblib.load(os.path.join(features_dir,"branch_comments_user_profiles_df_train.pkl"))
        branch_submission_dict_train = joblib.load(os.path.join(features_dir, "branch_submission_dict_train.pickle"))
        submission_data_dict_train = joblib.load(os.path.join(features_dir, "submission_data_dict_train.pickle"))
        branch_deltas_data_dict_train = joblib.load(os.path.join(features_dir, "branch_deltas_data_dict_train.pickle"))
        branches_lengths_list_train = joblib.load(os.path.join(features_dir, "branches_lengths_list_train.txt"))

        # load test data
        branch_comments_embedded_text_df_test = joblib.load(os.path.join(features_dir,"branch_comments_embedded_text_df_test.pkl"))
        branch_comments_features_df_test = joblib.load(os.path.join(features_dir,"branch_comments_features_df_test.pkl"))
        branch_comments_user_profiles_df_test = joblib.load(os.path.join(features_dir,"branch_comments_user_profiles_df_test.pkl"))
        branch_submission_dict_test = joblib.load(os.path.join(features_dir,"branch_submission_dict_test.pickle"))
        submission_data_dict_test = joblib.load(os.path.join(features_dir,"submission_data_dict_test.pickle"))
        branch_deltas_data_dict_test = joblib.load(os.path.join(features_dir,"branch_deltas_data_dict_test.pickle"))
        branches_lengths_list_test = joblib.load(os.path.join(features_dir,"branches_lengths_list_test.txt"))

        # debug only ---temp ----
        branch_comments_embedded_text_df_train = replace_0_with_list(branch_comments_embedded_text_df_train,
                                                                     len(branch_comments_embedded_text_df_train.loc[0,0]))
        branch_comments_features_df_train = replace_0_with_list(branch_comments_features_df_train,
                                                                     len(branch_comments_features_df_train.loc[0,0]))
        branch_comments_user_profiles_df_train = replace_0_with_list(branch_comments_user_profiles_df_train,
                                                                     len(branch_comments_user_profiles_df_train.loc[0,0]))

        branch_comments_embedded_text_df_test = replace_0_with_list(branch_comments_embedded_text_df_test,
                                                                     len(branch_comments_embedded_text_df_test.loc[0,0]))
        branch_comments_features_df_test = replace_0_with_list(branch_comments_features_df_test,
                                                                     len(branch_comments_features_df_test.loc[0,0]))
        branch_comments_user_profiles_df_test = replace_0_with_list(branch_comments_user_profiles_df_test,
                                                                     len(branch_comments_user_profiles_df_test.loc[0,0]))
        joblib.dump(branch_comments_embedded_text_df_train, 'branch_comments_embedded_text_df_train_debug.pkl')
        joblib.dump(branch_comments_features_df_train, 'branch_comments_features_df_train_debug.pkl')
        joblib.dump(branch_comments_user_profiles_df_train, 'branch_comments_user_profiles_df_train_debug.pkl')
        joblib.dump(branch_comments_embedded_text_df_test, 'branch_comments_embedded_text_df_test_debug.pkl')
        joblib.dump(branch_comments_features_df_test, 'branch_comments_features_df_test_debug.pkl')
        joblib.dump(branch_comments_user_profiles_df_test, 'branch_comments_user_profiles_df_test_debug.pkl')

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
    joblib.dump(train_model.measurements_dict, "measurements_dict.pkl")
    train_model.plot_loss(train_model.num_epochs, train_model.train_loss_list, train_model.test_loss_list)
    #
    # sys.stdout = old_stdout
    # log_file.close()


if __name__ == '__main__':
    main()

