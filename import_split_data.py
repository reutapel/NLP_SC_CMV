import pandas as pd
import os
from collections import defaultdict
import joblib


class ImportSplitData:
    """"This class implements the import, join and sort of all the data parts for each dataset """

    def __init__(self):

        self.folder_list = os.listdir(os.getcwd())
        self.data_folders_dict = defaultdict(list)
        self.all_data_dict = defaultdict(dict)

        self.collect_data_folders()
        self.load_join_data()

    def collect_data_folders(self):
        # get names of all data folders
        print("collecting all data folder names")
        for folder in self.folder_list:
            if folder[-6:-1] == 'train':
                self.data_folders_dict['train'].append(folder)
            elif folder[-6:-1] == 'testi':
                self.data_folders_dict['testi'].append(folder)
            elif folder[-6:-1] == 'valid':
                self.data_folders_dict['valid'].append(folder)

    def load_join_data(self):
        # collect all files by folder
        for dataset in self.data_folders_dict.keys():
            first_folder = True
            for folder in self.data_folders_dict[dataset]:
                # get folder path
                path = os.path.join(os.getcwd(), folder)
                # iterate over all files in each dataset folder
                for filename in os.listdir(path):
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
        # for each dataset reindex Dataframes by new joined & sorted len list
        for dataset in self.data_folders_dict.keys():
            len_dict = pd.DataFrame(data=self.all_data_dict[dataset]['branches_lengths_list_train'],
                                    index=self.all_data_dict[dataset]['branch_comments_embedded_text_df_train'].index)
            len_dict.sort_values(by=0, ascending=False, inplace=True)
            sorted_index_list = list(len_dict.index)
            for var in self.all_data_dict[dataset].keys():
                if isinstance(self.all_data_dict[dataset][var], pd.DataFrame):
                    self.all_data_dict[dataset][var] = self.all_data_dict[dataset][var].reindex(sorted_index_list)
                elif isinstance(self.all_data_dict[dataset][var], list):
                    self.all_data_dict[dataset][var] = len_dict[0].tolist()

        return self.all_data_dict