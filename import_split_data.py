import pandas as pd
import os
from collections import defaultdict
import joblib
from time import gmtime, strftime
import torchnet as tnt

class ImportSplitData:
    """"This class implements the import, join and sort of all the data parts in folders for each dataset, assuming
    numbered folders for train, testi, valid, with same list of objects inside """

    def __init__(self, cluster_dir=None):

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
