import pandas as pd
import os
from collections import defaultdict
import joblib
from time import gmtime, strftime
from utils import load_dataset_folders, get_model_layer_sizes, create_dataset_dict, define_model_hyper_params, load_data


class ImportSplitData:
    """"This class implements the import, join and sort of all the data parts in folders for each dataset, assuming
    numbered folders for train, testi, valid, with same list of objects inside """

    def __init__(self, cluster_dir=None):

        self.train_data = None
        self.test_data = None
        self.validation_data = None
        self.layers_input_size = None
        self.model_hyper_params_dict = None

        if cluster_dir is None:
            self.folder_list = os.listdir(os.path.join(os.getcwd(), 'features_to_use'))
        else:
            self.folder_list = os.listdir(os.path.join(os.getcwd(), 'clusters_features', cluster_dir))
        print(f'{strftime("%a, %d %b %Y %H:%M:%S", gmtime())} self.folder_list {self.folder_list}')

        self.data_folders_dict = defaultdict(list)
        self.all_data_dict = defaultdict(dict)

        self.collect_data_folders()

    def collect_data_folders(self):
        # get names of all data folders
        for folder in self.folder_list:
            if folder[0:5] == 'train':
                self.data_folders_dict['train'].append(folder)
            elif folder[0:5] == 'testi':
                self.data_folders_dict['testi'].append(folder)
            elif folder[0:5] == 'valid':
                self.data_folders_dict['valid'].append(folder)

        print(f'{strftime("%a, %d %b %Y %H:%M:%S", gmtime())} self.data_folders_dict {self.data_folders_dict}')

    def load_join_data(self):
        # collect all files by folder
        print(f'{strftime("%a, %d %b %Y %H:%M:%S", gmtime())} collect all files by folder')
        for dataset in ['train', 'testi']:  # self.data_folders_dict.keys():
            print(f'{strftime("%a, %d %b %Y %H:%M:%S", gmtime())} load from {dataset}')
            first_folder = True
            for folder in self.data_folders_dict[dataset]:
                print(f'{strftime("%a, %d %b %Y %H:%M:%S", gmtime())} load from {folder}')
                # get folder path
                path = os.path.join(os.getcwd(), 'features_to_use', folder)
                # iterate over all files in each dataset folder
                for filename in os.listdir(path):
                    print(f'{strftime("%a, %d %b %Y %H:%M:%S", gmtime())} load {filename}')
                    if filename == '.DS_Store':
                        continue
                    # connect all part of files of the same dataset
                    file_path = os.path.join(path, filename)
                    if first_folder:
                        self.all_data_dict[dataset][filename.split('.', 1)[0]] = joblib.load(file_path)
                    else:
                        file = joblib.load(file_path)
                        if isinstance(file, dict):
                            self.all_data_dict[dataset][filename.split('.', 1)[0]].update(file)
                        elif isinstance(file, list):
                            self.all_data_dict[dataset][filename.split('.', 1)[0]] = \
                                self.all_data_dict[dataset][filename.split('.', 1)[0]] + file
                        else:
                            self.all_data_dict[dataset][filename.split('.', 1)[0]] = \
                                self.all_data_dict[dataset][filename.split('.', 1)[0]].append(file,
                                                                                              verify_integrity=True)
                first_folder = False

    def sort_joined_data(self):
        # for each dataset train/testi/valid
        # 1. reindex Dataframes by new joined & sorted len list df
        # 2. replace dictionary keys in new order
        # 3. update len sorted list
        print(f'{strftime("%a, %d %b %Y %H:%M:%S", gmtime())} sort joined data')
        for dataset in self.data_folders_dict.keys():
            len_df = pd.DataFrame(data=self.all_data_dict[dataset]['branches_lengths_list'],
                                  index=self.all_data_dict[dataset]['branch_comments_embedded_text_df'].index)
            len_df.sort_values(by=0, ascending=False, inplace=True)
            sorted_index_list = list(len_df.index)
            self.all_data_dict[dataset]['len_df'] = len_df
            for var in self.all_data_dict[dataset].keys():
                if isinstance(self.all_data_dict[dataset][var], pd.DataFrame):
                    self.all_data_dict[dataset][var] = self.all_data_dict[dataset][var].reindex(sorted_index_list)
                elif isinstance(self.all_data_dict[dataset][var], list):
                    self.all_data_dict[dataset][var] = len_df[0].tolist()

        return self.all_data_dict

    def concat_datasets_strategy(self):
        """
        loads all data folders for train and test
        :return:
        """

        print('concat_datasets_strategy - populating train_data and test_data')
        # create pytorch dataset for each data folder and use torchnet.dataset.ConcatDataset to load data on the fly
        folders_data_dict = defaultdict(dict)
        # iterate all datasets
        for dataset, folder_list in self.data_folders_dict.items():
            # iterate all folders of dataset and load data
            folders_data_dict = load_dataset_folders(folders_data_dict, dataset, folder_list)

        self.train_data = folders_data_dict['train']
        self.test_data = folders_data_dict['testi']

        print('running get_model_layer_sizes')
        self.layers_input_size = get_model_layer_sizes(self.train_data[next(iter(self.train_data))])
        self.model_hyper_params_dict = define_model_hyper_params(self.layers_input_size)

    def import_all_strategy(self):

        print('import_all_strategy - populating train_data and test_data and validation_data')
        # join all data folders for each dataset approach
        self.load_join_data()
        all_data_dict = self.sort_joined_data()

        # load train data
        print(f'{strftime("%a, %d %b %Y %H:%M:%S", gmtime())} create train data')
        self.train_data = create_dataset_dict('train', all_data_dict)
        self.layers_input_size = get_model_layer_sizes(self.train_data)
        self.model_hyper_params_dict = define_model_hyper_params(self.layers_input_size)

        # load test data
        print(f'{strftime("%a, %d %b %Y %H:%M:%S", gmtime())} create test data')
        self.test_data = create_dataset_dict('test', all_data_dict)

        # load valid data
        if 'valid' in all_data_dict.keys():
            print(f'{strftime("%a, %d %b %Y %H:%M:%S", gmtime())} create validation data')
            self.validation_data = create_dataset_dict('valid', all_data_dict)

    def import_when_training_strategy(self):
        """
        load first folder just to determine layer sizes for model hyper parameters
        :return:
        """
        print('import_all_strategy - loading train0 to set data sizes')
        path = os.path.join(os.getcwd(), 'features_to_use', 'train0')
        # get dict of data objects per folder
        train0_data_dict = load_data(path)
        self.layers_input_size = get_model_layer_sizes(train0_data_dict)
        self.model_hyper_params_dict = define_model_hyper_params(self.layers_input_size)







