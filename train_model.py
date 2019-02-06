import torch as tr
import torch.nn as nn
from model_utils import CustomDataset
from torch.utils import data as dt
from DeltaModel import DeltaModel
from model_utils import InitLstm
from model_utils import InitConv1d
from import_split_data import ImportSplitData
import pandas as pd
from tqdm import tqdm
from time import gmtime, strftime
from sklearn import metrics
import joblib
import matplotlib.pyplot as plt
from pylab import savefig
from matplotlib.ticker import MaxNLocator
import os
import numbers
import sys
from datetime import datetime

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
                 input_size_sub_profile_features, fc1_droput, fc2_dropout, is_cuda, curr_model_outputs_dir):
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
        :param bool is_cuda: if running with cuda or not
        :param curr_model_outputs_dir: the directory to save the model's output
        """

        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.criterion = criterion
        self.is_cuda = is_cuda
        self.sigmoid = nn.Sigmoid()
        self.curr_model_outputs_dir = curr_model_outputs_dir

        # create model
        self.model = DeltaModel(init_lstm_text, init_lstm_comments, init_lstm_users, init_conv_text,
                                init_conv_sub_features, init_conv_sub_profile_features, input_size_text_sub,
                                input_size_sub_features, input_size_sub_profile_features, self.batch_size, num_labels,
                                fc1, fc2, fc1_droput, fc2_dropout, is_cuda)

        # on the model.parameters will be performed the update by stochastic gradient descent
        self.optimizer = tr.optim.Adam(self.model.parameters(), lr=learning_rate)

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

        if self.is_cuda:
            tr.backends.cudnn.benchmark = True

            self.model = self.model.cuda()
            print('run cuda')

        self.model.train()

        # device = tr.device("cuda" if tr.cuda.is_available() else "cpu")

        for epoch in tqdm(range(self.num_epochs), desc='epoch'):

            print("start epoch number ", epoch)

            correct = 0
            total = 0
            train_labels = tr.Tensor()
            train_predictions = tr.Tensor()
            train_probabilities = tr.Tensor()

            if self.is_cuda:
                train_labels = train_labels.cuda()
                train_predictions = train_predictions.cuda()
                train_probabilities = train_probabilities.cuda()

            first_batch = True
            for i, (data_points, labels) in tqdm(enumerate(self.train_loader), desc='batch'):

                # forward
                outputs, sorted_idx, batch_size = self.model(data_points)
                # TODO: understand impact of packed padded to loss, like function loss in model.py
                # turn to probability of class 1
                probabilities = self.sigmoid(outputs)
                outputs = outputs.view(batch_size, -1).float()

                # sort labels
                labels = labels[sorted_idx].view(batch_size, -1).float()
                if self.is_cuda:
                    labels = labels.cuda()
                    probabilities = probabilities.cuda()
                    outputs = outputs.cuda()

                # calculate for measurements
                predicted = ((probabilities > 0.5) * 1).float()
                total += labels.size(0)
                correct += (predicted == labels).sum()
                train_labels = tr.cat((train_labels, labels))
                train_predictions = tr.cat((train_predictions, predicted))
                train_probabilities = tr.cat((train_probabilities, probabilities))

                # calculate loss
                # print("calc loss")
                # pos_weight = tr.Tensor([(labels.sum() / (labels == 0).sum()).item(),
                #                         ((labels == 0).sum() / labels.sum()).item()])
                # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                # loss = criterion(outputs, labels)
                loss = self.criterion(outputs, labels)
                # if want graph per epoch
                if first_batch:
                    first_batch = False
                    self.train_loss_list.append(loss.item())
                else:
                    self.train_loss_list[epoch] += loss.item()

                # initialize gradient so only current batch will be summed and then backward
                self.optimizer.zero_grad()

                # calculate gradients
                loss.backward()

                # # debug gradients - self.model.parameters() contains every parameters that is defined in init
                # # and needs to have gradients, if something is defined in init and not used it's gradients will be 0
                # check gradients that are closed to input.. last in backwards
                # for p in self.model.parameters():
                #     print("gradient of p")
                #     print(p.grad.norm)

                # update parameters : tensor - learning_rate*gradient
                self.optimizer.step()

                if (i+1) % 100 == 0:
                    print('Epoch: [%d%d], Step: [%d%d], Loss: %.4f' % (epoch+1, self.num_epochs, i+1,
                                                                       len(self.train_dataset)//self.batch_size,
                                                                       loss))
            # average batch losses per epoch
            self.train_loss_list[epoch] = self.train_loss_list[epoch]/(i+1)
            # calculate measurements on train data
            self.calc_measurements(correct, total, train_labels, train_predictions, train_probabilities, epoch,  "train")
            self.test(epoch)
            self.model.train()

        # save the model
        tr.save(self.model.state_dict(), os.path.join(self.curr_model_outputs_dir, 'model.pkl'))

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
        test_probabilities = tr.Tensor()

        if self.is_cuda:
            test_labels = test_labels.cuda()
            test_predictions = test_predictions.cuda()
            test_probabilities = test_probabilities.cuda()

        # make sure no gradients - memory efficiency - no allocation of tensors to the gradients
        with(tr.no_grad()):

            for i, (data_points, labels) in tqdm(enumerate(self.test_loader), desc='batch'):

                outputs, sorted_idx, batch_size = self.model(data_points)

                # turn to probability of class 1
                probabilities = self.sigmoid(outputs)
                outputs = outputs.view(batch_size, -1).float()

                labels = labels[sorted_idx].view(batch_size, -1).float()

                if self.is_cuda:
                    outputs = outputs.cuda()
                    labels = labels.cuda()
                    probabilities = probabilities.cuda()

                # calculate loss
                # pos_weight = tr.Tensor([((labels == 0).sum() / labels.sum()).item(),
                #                         (labels.sum() / (labels == 0).sum()).item()])
                # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                # loss = criterion(outputs, labels)
                loss = self.criterion(outputs, labels)
                # if want graph per epoch
                if first_batch:
                    first_batch = False
                    self.test_loss_list.append(loss.item())
                else:
                    self.test_loss_list[epoch] += loss.item()

                predicted = (probabilities > 0.5).float() * 1
                total += labels.size(0)
                correct += (predicted == labels).sum()
                test_labels = tr.cat((test_labels, labels))
                test_predictions = tr.cat((test_predictions, predicted))
                test_probabilities = tr.cat((test_probabilities, probabilities))

        self.test_loss_list[epoch] = self.test_loss_list[epoch] / (i + 1)

        # calculate measurements on test data
        self.calc_measurements(correct, total, test_labels, test_predictions, test_probabilities, epoch, "test")

    def calc_measurements(self, correct, total, labels, pred, probabilities, epoch, dataset):

        if self.is_cuda:
            labels = labels.cpu()
            pred = pred.cpu()
            probabilities = probabilities.cpu()

        labels = labels.detach().numpy()
        pred = pred.detach().numpy()
        probabilities = probabilities.detach().numpy()
        print("calculate measurements on ", dataset)
        accuracy = float(correct) / float(total)
        print('Accuracy: ', accuracy)
        fpr, tpr, thresholds = metrics.roc_curve(labels, probabilities,
                                                 pos_label=1)
        auc = metrics.auc(fpr, tpr)
        print("AUC: ", auc)

        precision = metrics.precision_score(labels, pred)
        print("precision: ", precision)

        recall = metrics.recall_score(labels, pred)
        print("recall: ", recall)

        macro_f_score = metrics.f1_score(labels, pred, average='macro')
        print("macro_f_score: ", macro_f_score)

        self.measurements_dict[dataset][epoch] = [accuracy, auc, precision, recall, macro_f_score]

    def plot_graph(self, epoch_count, train_loss, test_loss, measurement):

        # Visualize history
        fig, ax = plt.subplots()
        ax.plot(list(range(epoch_count)), train_loss, 'g--', label='train')
        ax.plot(list(range(epoch_count)), test_loss, 'b-', label='test')
        ax.legend()
        plt.title(measurement + ' per epoch')
        plt.legend(['Train', 'Test'])

        # set ticks as int
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        # handle axis labels
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel(measurement, fontsize=12, rotation='horizontal', verticalalignment='bottom')
        plt.gca().yaxis.set_label_coords(0, 1.01)

        for i in ['top', 'right']:
            plt.gca().spines[i].set_visible(False)
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.05), ncol=1, frameon=True)  # bbox_to_anchor=(1.2, 1.05)
        # plt.legend(loc=2, bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=True)

        fig_to_save = fig
        fig_to_save.savefig(os.path.join(self.curr_model_outputs_dir, measurement+'_graph.png'))

    def plot_measurements(self, measurments_list=("accuracy", "auc", "precision", "recall", "macro_f_score")):

        measurements_dataset_df_dict = dict()

        for dataset in self.measurements_dict.keys():
            measurements_dataset_df_dict[dataset] = pd.DataFrame.from_dict(self.measurements_dict[dataset],
                                                                           orient='index')
            measurements_dataset_df_dict[dataset].columns = measurments_list
            joblib.dump(measurements_dataset_df_dict[dataset], os.path.join(self.curr_model_outputs_dir, dataset +
                                                                            '_measurements_dataset_df_dict.pkl'))

        for meas in measurments_list:
            for key in measurements_dataset_df_dict.keys():
                if key == 'train':
                    train_list = measurements_dataset_df_dict[key][meas].tolist()
                elif key == 'test':
                    test_list = measurements_dataset_df_dict[key][meas].tolist()
            self.plot_graph(self.num_epochs, train_list, test_list, meas)
        return


def replace_0_with_list(df, len_list_in_cell):
    for i, row in enumerate(df.values):
        for j, col in enumerate(row):
            if isinstance(col, numbers.Number):
                df.loc[i, j] = [0] * len_list_in_cell
    return df


def main(is_cuda):

    debug = False

    import_split_data_obj = ImportSplitData()
    all_data_dict = import_split_data_obj.sort_joined_data()

    if not debug:

        # load train data
        if 'train' in all_data_dict.keys():
            branch_comments_embedded_text_df_train = all_data_dict['train']['branch_comments_embedded_text_df_train']
            branch_comments_features_df_train = all_data_dict['train']['branch_comments_features_df_train']
            branch_comments_user_profiles_df_train = all_data_dict['train']['branch_comments_user_profiles_df_train']
            branch_submission_dict_train = all_data_dict['train']['branch_submission_dict_train']
            submission_data_dict_train = all_data_dict['train']['submission_data_dict_train']
            branch_deltas_data_dict_train = all_data_dict['train']['branch_deltas_data_dict_train']
            branches_lengths_list_train = all_data_dict['train']['branches_lengths_list_train']
            len_df_train = all_data_dict['train']['len_df']

        # load test data
        if 'testi' in all_data_dict.keys():
            branch_comments_embedded_text_df_test = all_data_dict['testi']['branch_comments_embedded_text_df_testi']
            branch_comments_features_df_test = all_data_dict['testi']['branch_comments_features_df_testi']
            branch_comments_user_profiles_df_test = all_data_dict['testi']['branch_comments_user_profiles_df_testi']
            branch_submission_dict_test = all_data_dict['testi']['branch_submission_dict_testi']
            submission_data_dict_test = all_data_dict['testi']['submission_data_dict_testi']
            branch_deltas_data_dict_test = all_data_dict['testi']['branch_deltas_data_dict_testi']
            branches_lengths_list_test = all_data_dict['testi']['branches_lengths_list_testi']
            len_df_test = all_data_dict['testi']['len_df']

        # load valid data
        if 'valid' in all_data_dict.keys():
            branch_comments_embedded_text_df_valid = all_data_dict['valid']['branch_comments_embedded_text_df_valid']
            branch_comments_features_df_valid = all_data_dict['valid']['branch_comments_features_df_valid']
            branch_comments_user_profiles_df_valid = all_data_dict['valid']['branch_comments_user_profiles_df_valid']
            branch_submission_dict_valid = all_data_dict['valid']['branch_submission_dict_valid']
            submission_data_dict_valid = all_data_dict['valid']['submission_data_dict_valid']
            branch_deltas_data_dict_valid = all_data_dict['valid']['branch_deltas_data_dict_valid']
            branches_lengths_list_valid = all_data_dict['valid']['branches_lengths_list_valid']
            len_df_valid = all_data_dict['valid']['len_df']

    else:
        # DEBUG
        # load train data
        text = {0: [[1., 2.], [3., 4.], [5., 6.]], 1: [[7., 8.], [9., 1.], [1., 5.]], 2: [[1., 4.], [1., 6.], [0., 0.]],
                3: [[9., 2.], [0., 0.], [0., 0.]]}
        comment_features = {0: [[10., 20., 34.], [30., 40., 54.], [50., 60., 70.]], 1: [[70., 80., 82.], [90., 10., 43.],
                                                                                         [11., 12., 65.]],
                            2: [[13., 14., 76.], [15., 16., 17.], [0., 0., 0.]],
                            3: [[19., 20., 65.], [0., 0., 0.], [0., 0., 0.]]}
        users_profiles = {0: [[111., 222.], [333., 444.], [555., 666.]], 1: [[777., 888.], [999., 100.], [111., 122.]],
                          2: [[133., 144.], [155., 166.], [0., 0.]],
                          3: [[190., 200.], [0., 0.], [0., 0.]]}
        branch_comments_embedded_text_df_train = pd.DataFrame(data=text)
        branch_comments_features_df_train = pd.DataFrame(data=comment_features)
        branch_comments_user_profiles_df_train = pd.DataFrame(data=users_profiles)
        branch_submission_dict_train = {0: ['6ax', [-4., 4., -44., 44.]], 1: ['6ax', [-5., 5., -55., 55.]], 2: ['3f9',
                                                                                                                [-2., 2., -22., 22.]]}
        submission_data_dict_train = {'6ax': [[12., 21.], [6., 9.], [1., 11., 111.]], '3f9': [[13., 31.], [66., 99.],
                                                                                              [1., 11., 111.]]}
        branch_deltas_data_dict_train = {0: [1., 2, [1, 3]], 1: [0., 0, []], 2: [1., 1, [1]]}
        branches_lengths_list_train = [4, 3, 2]

        # load test data
        branch_comments_embedded_text_df_test = pd.DataFrame(data=text)
        branch_comments_features_df_test = pd.DataFrame(data=comment_features)
        branch_comments_user_profiles_df_test = pd.DataFrame(data=users_profiles)
        branch_submission_dict_test = {0: ['6ax', [-4, 4, -44, 44]], 1: ['6ax', [-5, 5, -55, 55]], 2: ['3f9', [-2, 2,
                                                                                                               -22, 22]]}
        submission_data_dict_test = {'6ax': [[12, 21], [6, 9], [1, 11, 111]], '3f9': [[13, 31], [66, 99], [1, 11, 111]]}
        branch_deltas_data_dict_test = {0: [1, 2, ], 1: [0, 0, 0], 2: [1, 1, ]}
        branches_lengths_list_test = [4, 3, 2]

    train_data = branch_comments_embedded_text_df_train, branch_comments_features_df_train, \
                 branch_comments_user_profiles_df_train, branch_submission_dict_train, submission_data_dict_train, \
                 branch_deltas_data_dict_train, branches_lengths_list_train, len_df_train

    test_data = branch_comments_embedded_text_df_test, branch_comments_features_df_test, \
                branch_comments_user_profiles_df_test, branch_submission_dict_test, submission_data_dict_test, \
                branch_deltas_data_dict_test, branches_lengths_list_test, len_df_test

    # define hyper parameters of learning phase
    # log makes differences expand to higher numbers because of it's behaivor between 0 to 1
    criterion = nn.BCEWithLogitsLoss()
    learning_rate = 0.001  # range 0.0003-0.001 batch grows -> lr grows
    batch_size = 24  # TRY BATCH SIZE 100
    num_epochs = 2
    num_labels = 2
    fc1 = 32
    fc2 = 16
    fc1_dropout = 0.2
    fc2_dropout = 0.5

    base_directory = os.getenv('PWD')
    curr_model_outputs_dir = os.path.join(base_directory, 'model_outputs', datetime.now().strftime(
        f'%d_%m_%Y_%H_%M_LR_{learning_rate}_batch_size_{batch_size}_num_epochs_{num_epochs}_fc1_dropout_{fc1_dropout}_'
        f'fc2_dropout_{fc2_dropout}'))
    if not os.path.exists(curr_model_outputs_dir):
        os.makedirs(curr_model_outputs_dir)

    # define LSTM layers hyperparameters
    init_lstm_text = InitLstm(input_size=len(branch_comments_embedded_text_df_train.iloc[0, 0]), hidden_size=20,
                              num_layers=2, batch_first=True)
    init_lstm_comments = InitLstm(input_size=len(branch_comments_features_df_train.iloc[0, 0]), hidden_size=10,
                                  num_layers=2, batch_first=True)
    init_lstm_users = InitLstm(input_size=len(branch_comments_user_profiles_df_train.iloc[0, 0]), hidden_size=10,
                               num_layers=2, batch_first=True)

    # define conv layers hyperparameters
    init_conv_text = InitConv1d(in_channels=1, out_channels=9, kernel_size=3, stride=1, padding=0,
                                leaky_relu_alpha=0.2)
    init_conv_sub_features = InitConv1d(in_channels=1, out_channels=9, kernel_size=3, stride=1, padding=0,
                                        leaky_relu_alpha=0.2)
    init_conv_sub_profile_features = InitConv1d(in_channels=1, out_channels=9, kernel_size=3, stride=1, padding=0,
                                                leaky_relu_alpha=0.2)

    input_size_text_sub = len(branch_comments_embedded_text_df_train.iloc[0, 0])
    input_size_sub_features = len(submission_data_dict_train[list(submission_data_dict_train.keys())[0]][1]) + \
                              len(branch_submission_dict_train[list(branch_submission_dict_train.keys())[0]][1])
                                # submission features + branch features
    input_size_sub_profile_features = len(submission_data_dict_train[list(submission_data_dict_train.keys())[0]][2])

    # create training instance
    train_model = TrainModel(train_data, test_data, learning_rate, criterion, batch_size, num_epochs, num_labels, fc1,
                             fc2, init_lstm_text, init_lstm_comments, init_lstm_users, init_conv_text,
                             init_conv_sub_features, init_conv_sub_profile_features, input_size_text_sub,
                             input_size_sub_features, input_size_sub_profile_features, fc1_dropout, fc2_dropout,
                             is_cuda, curr_model_outputs_dir)

    # train and test model
    train_model.train()
    joblib.dump(train_model.measurements_dict, os.path.join(curr_model_outputs_dir, 'measurements_dict.pkl'))

    measurments_list = ["accuracy", "auc", "precision", "recall", 'macro_f_score']
    train_model.plot_graph(train_model.num_epochs, train_model.train_loss_list, train_model.test_loss_list, 'loss')
    train_model.plot_measurements(measurments_list)

    # sys.stdout = old_stdout
    # log_file.close()


if __name__ == '__main__':
    main_is_cuda = sys.argv[1]
    print(f'running with cuda: {main_is_cuda}')
    if main_is_cuda == 'False':
        main_is_cuda = False
    main(main_is_cuda)
