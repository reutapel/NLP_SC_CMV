from time import gmtime, strftime
import os
import torch as tr
import math
import numbers
from model_utils import CustomDataset, InitLstm, InitConv1d
from torch.utils import data as dt
import joblib
import pandas as pd


def load_data(folder_path: str) -> dict:
    """
    Load data from a specific folder
    :param folder_path: str: the folder's path to load from
    :return: dict: {file_name: file} with all the data in the folder
    """
    # iterate over all files in each dataset folder
    data_dict = dict()
    for filename in os.listdir(folder_path):
        print(f'{strftime("%a, %d %b %Y %H:%M:%S", gmtime())} load {filename} from {folder_path}')
        if filename == '.DS_Store':
            continue
        # connect all part of files of the same dataset
        file_path = os.path.join(folder_path, filename)
        file = joblib.load(file_path)
        data_dict[filename.split('.', 1)[0]] = file

    len_df = pd.DataFrame(data=data_dict[f'branches_lengths_list'],
                          index=data_dict[f'branch_comments_embedded_text_df'].index)
    data_dict['len_df'] = len_df

    return data_dict


def create_class_weight(labels: tr.Tensor, mu=None):
    """
    This function create weight tensor for loss function
    :param labels: tensor with the labels of the batch
    :param mu: parameter to tune
    :return:
    """

    if mu is None:
        mu = 1.0

    sizes = {'delta': labels.sum().item(),
             'no_delta': (labels == 0).sum().item()}
    total = sum(sizes.values())
    weights = dict()

    for label, count_labels in sizes.items():
        if count_labels > 0:
            w = total / float(count_labels)
        else:
            w = total
        score = math.log2(mu*w)
        weights[label] = max(score, 1.0)

    weight = tr.where(labels == 0, (labels == 0).float() * weights['delta'],
                      (labels == 1).float() * weights['no_delta'])

    return weight


def create_dataset_dict(dataset: str, all_data_dict: dict) -> dict:
    """
    The function get dataset name and dict and return a dict for this dataset
    :param dataset: the name of the data set (train, testi, valid)
    :param all_data_dict: the dict of all the data
    :return:
    """

    return {
        'branch_comments_embedded_text_df': all_data_dict[dataset][f'branch_comments_embedded_text_df'],
        'branch_comments_features_df': all_data_dict[dataset][f'branch_comments_features_df'],
        'branch_comments_user_profiles_df': all_data_dict[dataset][f'branch_comments_user_profiles_df'],
        'branch_submission_dict': all_data_dict[dataset][f'branch_submission_dict'],
        'submission_data_dict': all_data_dict[dataset][f'submission_data_dict'],
        'branch_deltas_data_dict': all_data_dict[dataset][f'branch_deltas_data_dict'],
        'branches_lengths_list': all_data_dict[dataset][f'branches_lengths_list'],
        'len_df': all_data_dict[dataset]['len_df']
    }


def get_model_layer_sizes(data_dict: dict) -> dict:
    """
    The function get dict with the data and return the sizes for the model's layers
    :param data_dict:
    :return:
    """

    return {
        'lstm_text': len(data_dict['branch_comments_embedded_text_df'].iloc[0, 0]),
        'lstm_comments': len(data_dict['branch_comments_features_df'].iloc[0, 0]),
        'lstm_users': len(data_dict['branch_comments_user_profiles_df'].iloc[0, 0]),
        'input_size_text_sub': len(data_dict['branch_comments_embedded_text_df'].iloc[0, 0]),
        'input_size_sub_features':
            len(data_dict['submission_data_dict'][list(data_dict['submission_data_dict'].keys())[0]][1]) +
            len(data_dict['branch_submission_dict'][list(data_dict['branch_submission_dict'].keys())[0]][1]),
        'input_size_sub_profile_features':
            len(data_dict['submission_data_dict'][list(data_dict['submission_data_dict'].keys())[0]][2])

    }


def load_dataset_folders(def_dict, dataset: str, folder_list: list):
    """

    :param def_dict: default dict {dataset: {folder_name : data dict}}
    :param dataset: dataset name e.g. train/testi/valid
    :param folder_list: folder name list of dataset
    :return: populated def_dict
    """
    for folder in folder_list:
        print(f'{strftime("%a, %d %b %Y %H:%M:%S", gmtime())} load from {folder}')
        # get folder path
        path = os.path.join(os.getcwd(), 'features_to_use', folder)
        # get dict of data objects per folder
        def_dict[dataset][folder] = load_data(path)

    return def_dict


def create_dataset(data):
    """

    :param data: all the data structures needed for the class
    :return: CustomDataset object
    """

    return CustomDataset(data)


def create_data_loader(dataset, batch_size):
    """

    :param dataset: dataset train or test for data loader to run over
    :param batch_size: size of batch
    :return: data loader object
    """

    return dt.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


def create_dataset_data_loader(folder, batch_size):
    print(f'{strftime("%a, %d %b %Y %H:%M:%S", gmtime())} in create_dataset_data_loader load from {folder}')
    # get folder path
    path = os.path.join(os.getcwd(), 'features_to_use', folder)
    # get dict of data objects per folder
    data_dict = load_data(path)
    # create dataset
    dataset = create_dataset(data_dict)
    # create data loader
    loader = create_data_loader(dataset, batch_size)
    return dataset, loader


def replace_0_with_list(df, len_list_in_cell):
    for i, row in enumerate(df.values):
        for j, col in enumerate(row):
            if isinstance(col, numbers.Number):
                df.loc[i, j] = [0] * len_list_in_cell
    return df


def define_model_hyper_params(layers_input_size):
    # define LSTM layers hyperparameters
    hyperparams_dict = dict()
    # TODO: define sizes according to concat dataset flag
    print(f'{strftime("%a, %d %b %Y %H:%M:%S", gmtime())} define LSTM layers hyperparameters')
    hyperparams_dict['init_lstm_text'] = InitLstm(input_size=layers_input_size['lstm_text'], hidden_size=20, num_layers=2, batch_first=True)
    hyperparams_dict['init_lstm_comments'] = InitLstm(input_size=layers_input_size['lstm_comments'], hidden_size=10, num_layers=2,
                                  batch_first=True)
    hyperparams_dict['init_lstm_users'] = InitLstm(input_size=layers_input_size['lstm_users'], hidden_size=10, num_layers=2,
                               batch_first=True)

    # define conv layers hyperparameters
    print(f'{strftime("%a, %d %b %Y %H:%M:%S", gmtime())} define conv layers hyperparameters')
    hyperparams_dict['init_conv1d_text'] = InitConv1d(in_channels=1, out_channels=9, kernel_size=3, stride=1, padding=0,
                                leaky_relu_alpha=0.2)
    hyperparams_dict['init_conv1d_sub_features'] = InitConv1d(in_channels=1, out_channels=9, kernel_size=3, stride=1, padding=0,
                                        leaky_relu_alpha=0.2)
    hyperparams_dict['init_conv1d_sub_profile_features'] = InitConv1d(in_channels=1, out_channels=9, kernel_size=3, stride=1, padding=0,
                                                leaky_relu_alpha=0.2)

    hyperparams_dict['input_size_text_sub'] = layers_input_size['input_size_text_sub']
    hyperparams_dict['input_size_sub_features'] = layers_input_size['input_size_sub_features']
    hyperparams_dict['input_size_sub_profile_features'] = layers_input_size['input_size_sub_profile_features']
    return hyperparams_dict
