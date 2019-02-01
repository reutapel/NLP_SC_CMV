import joblib
import pandas as pd


import torch as tr
import torch.nn as nn
from sklearn import metrics

trt = tr.Tensor([[-14,-12,-0.2]])
sigmoid = nn.Sigmoid()
fpr, tpr, thresholds = metrics.roc_curve([0, 1, 1], sigmoid(trt),
                                         pos_label=1)
auc = metrics.auc(fpr, tpr)


#
# measurements_dict = joblib.load("measurements_dict.pkl")
#
# print("hi")
#
# def plot_measurements(self, measurments_list=["accuracy", "auc", "precision", "recall"]):
#     measurements_dataset_df_dict = dict()
#     for dataset in self.measurements_dict.keys():
#         measurements_dataset_df_dict[dataset] = pd.DataFrame.from_dict(self.measurements_dict[dataset], orient='index')
#         measurements_dataset_df_dict[dataset].columns = measurments_list
#         joblib.dump(measurements_dataset_df_dict[dataset], dataset+'_measurements_dataset_df_dict.pkl')
#
#     for meas in measurments_list:
#         train_list = measurements_dataset_df_dict['train'][meas].tolist()
#         test_list = measurements_dataset_df_dict['test'][meas].tolist()
#         self.plot_graph(self.num_epochs, train_list, test_list, meas)
#     return