import torch as tr
import torch.nn as nn
from DeltaModel import DeltaModel
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
import sys
from datetime import datetime
from collections import defaultdict
from early_stopping_pytorch.pytorchtools import EarlyStopping
from utils import create_class_weight, create_dataset_data_loader
# #!/usr/bin/env python
# import psutil
# # you can convert that object to a dictionary
# print(dict(psutil.virtual_memory()._asdict()))


# old_stdout = sys.stdout
# log_file = open("train_model.log", "w")
# sys.stdout = log_file


# TODO: F.mse_loss(size_average, reduce) : parameters that affect if we get average values per batch : sum or average


class TrainModel:
    """
    class builds the data sets, data loaders, model, trains and tests the model.
    """
    def __init__(self, import_split_data_obj, learning_rate, criterion, batch_size, num_epochs,
                 num_labels, fc1, fc2, fc1_droput, fc2_dropout, is_cuda, curr_model_outputs_dir,
                 concat_datasets=False, average_loss_per_batch=True):
        """
        :param import_split_data_obj: ImportSplitData object - holds info of data directory structure,
        and train/test if pre loaded
        :param learning_rate: learning rate pace
        :param criterion: which criterion to calculate loss by
        :param batch_size: size of batch
        :param num_epochs: number of epochs to train
        :param num_labels: number of labels in data
        :param fc1: first linear reduction size from concatenated hidden size to
        :param fc2: second reduction , before reduction to label dimension of 2
        :param fc1_droput: probability for first dropout
        :param fc2_dropout: probability for second dropout
        :param bool is_cuda: if running with cuda or not
        :param curr_model_outputs_dir: the directory to save the model's output
        :param average_loss_per_batch: if to average epoch loss in numbers of batches - so plot loss of avg batch and
        not epoch
        """
        self.import_split_data_obj = import_split_data_obj
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.criterion = criterion
        self.is_cuda = is_cuda
        self.sigmoid = nn.Sigmoid()
        self.curr_model_outputs_dir = curr_model_outputs_dir
        self.average_loss_per_batch = average_loss_per_batch
        # create model
        self.model = DeltaModel(import_split_data_obj.model_hyper_params_dict, self.batch_size, num_labels,
                                fc1, fc2, fc1_droput, fc2_dropout, is_cuda)

        # on the model.parameters will be performed the update by stochastic gradient descent
        self.optimizer = tr.optim.Adam(self.model.parameters(), lr=learning_rate)
        # TODO - change to fit import strategy
        # # create datasets
        # if concat_datasets:
        #     train_datasets_list = list()
        #     test_datasets_list = list()
        #     # create customdataset per folder for train
        #     for folder, data_dict in import_split_data_obj.train_data.items():
        #         folder_train_dataset = create_dataset(data_dict)
        #         train_datasets_list.append(folder_train_dataset)
        #     # concatenate datasets for on the fly loading of data
        #     self.train_dataset = torchnet.dataset.ConcatDataset(train_datasets_list)
        #     # create customdataset per folder for test
        #     for folder, data_dict in import_split_data_obj.test_data.items():
        #         folder_test_dataset = create_dataset(data_dict)
        #         test_datasets_list.append(folder_test_dataset)
        #     # concatenate datasets for on the fly loading of data
        #     self.test_dataset = torchnet.dataset.ConcatDataset(test_datasets_list)
        #
        # else:
        #     self.train_dataset = create_dataset(import_split_data_obj.train_data)
        #     self.test_dataset = create_dataset(import_split_data_obj.test_data)
        #
        # # create data loaders
        # self.train_loader = create_data_loader(self.train_dataset, self.batch_size)
        # self.test_loader = create_data_loader(self.test_dataset, self.batch_size)

        self.train_loss_list = list()
        self.test_loss_list = list()
        self.mu_train_loss_dict = defaultdict(list)
        self.mu_test_loss_dict = defaultdict(list)
        self.measurements_dict = dict()
        self.measurements_dict["train"] = dict()
        self.measurements_dict["test"] = dict()

        self.train_dataset = None
        self.train_loader = None

        self.test_dataset = None
        self.test_loader = None

        # calculate number of trainable parameters
        print(sum(p.numel() for p in self.model.parameters() if p.requires_grad), " trainable parameters in model")

    def train(self, mu=None, patience=10):
        """
        train the model on train data by batches iterate all epochs
        :param: float mu: mu for creating weight in loss
        :return:
        """

        if self.is_cuda:
            tr.backends.cudnn.benchmark = True

            self.model = self.model.cuda()
            print('run cuda')

        self.model.train()

        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=patience, verbose=True)

        # device = tr.device("cuda" if tr.cuda.is_available() else "cpu")

        for epoch in tqdm(range(self.num_epochs), desc='epoch'):

            print("start epoch number ", epoch)

            correct = 0
            total = 0
            train_labels = tr.Tensor()
            train_predictions = tr.Tensor()
            train_probabilities = tr.Tensor()
            batch_cnt = 0
            if self.is_cuda:
                train_labels = train_labels.cuda()
                train_predictions = train_predictions.cuda()
                train_probabilities = train_probabilities.cuda()

            first_folder = True
            # loop on train folders, create dataset / data loader for each folder and train every epoch on all the
            # folders sequentially
            # create dataset and data loader
            print('training by folders') # TODO: SHIMON - GENERALIZE TO ALL IMPORT STRATEGIES?
            for folder in self.import_split_data_obj.data_folders_dict['train']:
                print(" loading train folder: ", folder)
                self.train_dataset = None
                self.train_loader = None
                self.train_dataset, self.train_loader = create_dataset_data_loader(folder, self.batch_size)

                for i, (data_points, labels) in tqdm(enumerate(self.train_loader), desc='batch'):

                    # forward
                    outputs, sorted_idx, batch_size = self.model(data_points)

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
                    weights = create_class_weight(labels, mu)
                    criterion = nn.BCEWithLogitsLoss(weight=weights)
                    loss = criterion(outputs, labels)

                    # loss = self.criterion(outputs, labels)
                    # if bool(criterion(outputs, labels) != self.criterion(outputs, labels)):
                    #     print(f'not the same loss with and without pos_weight for epoch {epoch} and step {i}')
                    # loss = self.criterion(outputs, labels)
                    # we want loss graph per epoch so we sum loss of all batches per epoch
                    if first_folder:
                        first_folder = False
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

                    if (i+1) % 20 == 0:
                        print('Epoch: [%d of %d], Step: [%d of %d], Loss: %.4f' % (epoch+1, self.num_epochs, i+1,
                                                                           len(self.train_dataset)//self.batch_size,
                                                                           loss))
                print("finished folder %s iteration" % folder)
                batch_cnt += i
                print("number of batches from all trained folders so far is: ", str(batch_cnt))
            print("finished all train folders for epoch: ", str(epoch))
            if self.average_loss_per_batch:
                # average batch losses per epoch
                self.train_loss_list[epoch] = self.train_loss_list[epoch]/(batch_cnt+1)

            # calculate measurements on train data
            self.calc_measurements(correct, total, train_labels, train_predictions, train_probabilities, epoch, "train")
            self.test(epoch)

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(self.test_loss_list[epoch], self.model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

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
            first_folder = True
            batch_cnt = 0
            # average batch losses per epoch
            # loop on test folders, create dataset / data loader for each folder and test every epoch on all the
            # folders sequentially
            # create dataset and data loader
            print('testing by folders')        # TODO: SHIMON - GENERALIZE TO ALL IMPORT STRATEGIES?
            for folder in self.import_split_data_obj.data_folders_dict['testi']:
                self.test_dataset = None
                self.test_loader = None
                self.test_dataset, self.test_loader = create_dataset_data_loader(folder, self.batch_size)

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
                    # pos_weight_delta = ((labels.long() == 0).sum() / labels.long().sum()).item()
                    # pos_weight_no_delta = (labels.long().sum() / (labels.long() == 0).sum()).item()
                    #
                    # pos_weight = tr.where(labels == 0, (labels == 0).float()*pos_weight_no_delta,
                    #                       (labels == 1).float()*pos_weight_delta)
                    # criterion = nn.BCEWithLogitsLoss(weight=pos_weight)
                    weights = create_class_weight(labels)
                    criterion = nn.BCEWithLogitsLoss(weight=weights)
                    loss = criterion(outputs, labels)
                    # loss = self.criterion(outputs, labels)
                    # if want graph per epoch
                    if first_folder:
                        first_folder = False
                        self.test_loss_list.append(loss.item())
                    else:
                        self.test_loss_list[epoch] += loss.item()

                    predicted = (probabilities > 0.5).float() * 1
                    total += labels.size(0)
                    correct += (predicted == labels).sum()
                    test_labels = tr.cat((test_labels, labels))
                    test_predictions = tr.cat((test_predictions, predicted))
                    test_probabilities = tr.cat((test_probabilities, probabilities))
                batch_cnt += i
        if self.average_loss_per_batch:
            # average batch losses per epoch
            self.test_loss_list[epoch] = self.test_loss_list[epoch] / (batch_cnt + 1)

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

        macro_precision = metrics.precision_score(labels, pred, average='macro')
        print("Macro precision: ", macro_precision)

        macro_recall = metrics.recall_score(labels, pred , average='macro')
        print("Macro recall: ", macro_recall)

        macro_f_score = metrics.f1_score(labels, pred, average='macro')
        print("Macro f_score: ", macro_f_score)

        precision = metrics.precision_score(labels, pred)
        print("precision: ", precision)

        recall = metrics.recall_score(labels, pred)
        print("recall: ", recall)

        f_score = metrics.f1_score(labels, pred)
        print("f_score: ", f_score)

        self.measurements_dict[dataset][epoch] = [accuracy, auc, precision, recall, f_score,
                                                  macro_precision, macro_recall, macro_f_score]

    def plot_graph(self, epoch_count, train_loss, test_loss, measurement):

        # Visualize history
        fig, ax = plt.subplots()
        ax.plot(list(range(epoch_count)), train_loss, 'g--', label='train')
        ax.plot(list(range(epoch_count)), test_loss, 'b-', label='test')

        # find position of lowest validation loss
        minposs = test_loss.index(min(test_loss)) + 1
        plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

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

    def plot_measurements(self, measurments_list=("accuracy", "auc", "precision", "recall", "f_score",
                                                  "macro_precision", "macro_recall", "macro_f_score")):

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


def main(is_cuda, cluster_dir=None):
    #TODO: REUT - UNDERSTAND IF import strategy and training affects mu or clusters
    concat_datasets = False
    import_data_strategy_dict = {0: 'import_all', 1: 'concat_datasets', 2: 'import_when_training'}
    chosen_import_strategy = import_data_strategy_dict[2]
    print('chosen import strategy is: ', chosen_import_strategy)

    # create import obj and collect names of data folders
    import_split_data_obj = ImportSplitData()
    if chosen_import_strategy == 'concat_datasets':
        import_split_data_obj.concat_datasets_strategy()

    elif chosen_import_strategy == 'import_all':
        import_split_data_obj.import_all_strategy()
    else:
        print('import_when_training - will be dealt in model.train()')
        import_split_data_obj.import_when_training_strategy()

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

    base_directory = os.getcwd() # os.getenv('PWD')
    print(f'{strftime("%a, %d %b %Y %H:%M:%S", gmtime())} create outputs directory')
    curr_model_outputs_dir = os.path.join(base_directory, 'model_outputs', datetime.now().strftime(
        f'%d_%m_%Y_%H_%M_LR_{learning_rate}_batch_size_{batch_size}_num_epochs_{num_epochs}_fc1_dropout_{fc1_dropout}_'
        f'fc2_dropout_{fc2_dropout}'))
    if not os.path.exists(curr_model_outputs_dir):
        os.makedirs(curr_model_outputs_dir)

    # create training instance
    print(f'{strftime("%a, %d %b %Y %H:%M:%S", gmtime())} create training instance')
    train_model = TrainModel(import_split_data_obj, learning_rate, criterion, batch_size, num_epochs, num_labels, fc1,
                             fc2, fc1_dropout, fc2_dropout,
                             is_cuda, curr_model_outputs_dir, concat_datasets=concat_datasets)

    # train and test model
    print(f'{strftime("%a, %d %b %Y %H:%M:%S", gmtime())} train and test model')
    for curr_mu in [1.5, 2.0, 2.5, 3.0]:
        print(f'{strftime("%a, %d %b %Y %H:%M:%S", gmtime())} train for mu: {curr_mu}')
        # initialize measures lists and dicts
        train_model.train_loss_list = list()
        train_model.test_loss_list = list()
        train_model.measurements_dict = dict()
        train_model.measurements_dict["train"] = dict()
        train_model.measurements_dict["test"] = dict()

        curr_model_outputs_mu_dir = os.path.join(curr_model_outputs_dir, f'mu_{curr_mu}')
        if not os.path.exists(curr_model_outputs_mu_dir):
            os.makedirs(curr_model_outputs_mu_dir)
        train_model.curr_model_outputs_dir = curr_model_outputs_mu_dir
        print(" starting train")
        train_model.train(mu=curr_mu)
        joblib.dump(train_model.measurements_dict, os.path.join(curr_model_outputs_dir, 'measurements_dict.pkl'))

        print(f'{strftime("%a, %d %b %Y %H:%M:%S", gmtime())} plot graphs')
        measurments_list = ["accuracy", "auc", "precision", "recall", "f_score", "macro_precision", "macro_recall",
                            "macro_f_score"]
        print("plotting graph")
        train_model.plot_graph(train_model.num_epochs, train_model.train_loss_list, train_model.test_loss_list, 'loss')
        train_model.mu_train_loss_dict[curr_mu] = train_model.train_loss_list
        train_model.mu_test_loss_dict[curr_mu] = train_model.test_loss_list
        print("plotting measurements")
        train_model.plot_measurements(measurments_list)

    joblib.dump(train_model.mu_train_loss_dict, os.path.join(curr_model_outputs_dir, 'mu_train_loss_dict.pkl'))
    joblib.dump(train_model.mu_test_loss_dict, os.path.join(curr_model_outputs_dir, 'mu_test_loss_dict.pkl'))

    # sys.stdout = old_stdout
    # log_file.close()


if __name__ == '__main__':
    main(is_cuda=False)
    # """
    # sys.argv[1] = main_is_cuda
    # sys.argv[2] = cluster directory name
    # """
    # main_is_cuda = sys.argv[1]
    # if len(sys.argv) > 2:
    #     cluster_directory = sys.argv[2]
    # else:
    #     cluster_directory = None
    # print(f'running with cuda: {main_is_cuda}')
    # if main_is_cuda == 'False':
    #     main_is_cuda = False
    # main(main_is_cuda, cluster_directory)
    # print(f'{strftime("%a, %d %b %Y %H:%M:%S", gmtime())} Done!')
