import torch as tr
import pandas as pd
import numpy as np
import torch
import numbers
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# # # #
# # # # ################################################ TENSOR GRADIENTS #####################################################
# # # # ######################################################################################################################
# # # x = torch.randn(2, 2)
# # # y = torch.randn(2, 2)
# # # # By default, user created Tensors have ``requires_grad=False``
# # # print(x.requires_grad, y.requires_grad)
# # # z = x + y
# # # # So you can't backprop through z
# # # print(z.grad_fn)
# # #
# # # # ``.requires_grad_( ... )`` changes an existing Tensor's ``requires_grad``
# # # # flag in-place. The input flag defaults to ``True`` if not given.
# # # x = x.requires_grad_()
# # # y = y.requires_grad_()
# # # # z contains enough information to compute gradients, as we saw above
# # # z = x + y
# # # print(z.grad_fn)
# # # # If any input to an operation has ``requires_grad=True``, so will the output
# # # print(z.requires_grad)
# # #
# # # # Now z has the computation history that relates itself to x and y
# # # # Can we just take its values, and **detach** it from its history?
# # # new_z = z.detach()
# # #
# # # # ... does new_z have information to backprop to x and y?
# # # # NO!
# # # print(new_z.grad_fn)
# # # # And how could it? ``z.detach()`` returns a tensor that shares the same storage
# # # # as ``z``, but with the computation history forgotten. It doesn't know anything
# # # # about how it was computed.
# # # # In essence, we have broken the Tensor away from its past history
# # #
# # # # You can also stop autograd from tracking history on Tensors with .requires_grad``=True by wrapping the code block in
# # # # ``with torch.no_grad():
# # #
# # # print(x.requires_grad)
# # # print((x ** 2).requires_grad)
# # #
# # # with torch.no_grad():
# # #     print((x ** 2).requires_grad)
# # #
# # # ######################################################################################################################
# # # ######################################################################################################################
# # #
# # #
# # # ############## RELU ###############
# # #
# # # data = torch.randn(2, 2)
# # # print(data)
# # # print(F.relu(data))
# # #
# # # ###################################a
# # #
# # #
# # # ############## SOFTMAX ###############
# # # # just a non-linearity, but it is special in that it usually is the last operation done in a network.
# # # # This is because it takes in a vector of real numbers and returns a probability distribution
# # # # Softmax is also in torch.nn.functional
# # # data = torch.randn(5)
# # # print(data)
# # # print(F.softmax(data, dim=0))
# # # print(F.softmax(data, dim=0).sum())  # Sums to 1 because it is a distribution!
# # # print(F.log_softmax(data, dim=0))  # theres also log_softmax
# # #
# # # ####################################
# # #
# # #
# # #
# # # T_data = [[[1., 2.], [3., 4.]],
# # #           [[5., 6.], [7., 8.]]]
# # # T = torch.tensor(T_data, requires_grad=True)
# # # print(T[0,1,1].item())
# # # print(T.grad_fn)
# # #
# # # # Tensor factory methods have a ``requires_grad`` flag
# # # x = torch.tensor([1., 2., 3], requires_grad=True)
# # #
# # # # With requires_grad=True, you can still do all the operations you previously
# # # # could
# # # y = torch.tensor([4., 5., 6], requires_grad=True)
# # # z = x + y
# # # print(z)
# # #
# # # # BUT z knows something extra.
# # # print(z.grad_fn)
# # #
# # # # tr.manual_seed(1)
# # # # a = tr.rand(3,4,5)
# # # #
# # # # print(tr.__version__)
# # #
# # # # rnn = tr.nn.LSTMCell(10, 20)
# # # # print("rnn is", rnn)
# # # # input = tr.randn(6, 3, 10)
# # # # print("input is", input)
# # # # hx = tr.randn(3, 20)
# # # # print("hx is", hx)
# # # # cx = tr.randn(3, 20)
# # # # print("cx is", cx)
# # # # output = []
# # # # for i in range(6):
# # # #     hx, cx = rnn(input[i], (hx, cx))
# # # #     print("hx is", hx)
# # # #     print("cx is", cx)
# # # #     output.append(hx)
# # # # #     print("output is", output)
# # # #
# # # # raw_text_df = pd.Series([['sdfsdfsd'],['sdfsg']])
# # # # raw_text_tensor = tr.Tensor(raw_text_df.values)
# # # # print(raw_text_tensor)
# # #
# # #
# # # # lstm = nn.LSTM(3, 3)  # Input dim is 3, output dim is 3
# # # # inputs = [torch.randn(1, 3) for _ in range(5)]  # make a sequence of length 5
# # # # print(inputs)
# # #
# # # # csv = "all_submissions_comments_with_label_all_deltalog_final.csv"
# # # # comments_with_label = pd.read_csv(csv)
# # # # all_submissions_comments_with_label_all_deltalog_final_comment_body = comments_with_label.comment_body
# # # # all_submissions_comments_with_label_all_deltalog_final_comment_body = \
# # # #     all_submissions_comments_with_label_all_deltalog_final_comment_body.str.lstrip('b"')
# # # # all_submissions_comments_with_label_all_deltalog_final_comment_body = \
# # # #     all_submissions_comments_with_label_all_deltalog_final_comment_body.str.lstrip("b'")
# # # # # all_submissions_comments_with_label_all_deltalog_final_comment_body = \
# # # # #     all_submissions_comments_with_label_all_deltalog_final_comment_body[all_submissions_comments_with_label_all_deltalog_final_comment_body.str.contains("Confirmed") == False]
# # # # # all_submissions_comments_with_label_all_deltalog_final_comment_body = \
# # # # #     all_submissions_comments_with_label_all_deltalog_final_comment_body[all_submissions_comments_with_label_all_deltalog_final_comment_body.str.contains("[deleted]") == False]
# # # # all_submissions_comments_with_label_all_deltalog_final_comment_body.to_csv("all_submissions_comments_with_label_all_deltalog_final_comment_body.csv")
# # # # print("Eli")
# # #
# # # # def evaluate_self_sim(self):
# # # #     """
# # # #     sanity check: method checks if the model makes a doc most similar to itself
# # # #     method prints most, second most, median and least similar vec to a doc
# # # #     :return:
# # # #     """
# # # #
# # # #     ranks = []
# # # #     second_ranks = []
# # # #     doc_iter = 0
# # # #     for doc_id in range(len(self.train_corpus)):
# # # #         inferred_vector = self.doc2vec_model.infer_vector(self.train_corpus[doc_id].words)
# # # #         sims = self.doc2vec_model.docvecs.most_similar([inferred_vector], topn=len(self.doc2vec_model.docvecs))
# # # #         rank = [docid for docid, sim in sims].index(doc_id)
# # # #         ranks.append(rank)
# # # #         second_ranks.append(sims[1])
# # # #         if doc_iter % 500000 == 0:
# # # #             print('Document ({}): «{}»\n'.format(doc_id, ' '.join(self.train_corpus[doc_id].words)))
# # # #             print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % self.doc2vec_model)
# # # #             for label, index in [('MOST', 0), ('SECOND-MOST', 1),
# # # #                                  ('MEDIAN', len(sims) // 2), ('LEAST', len(sims) - 1)]:
# # # #                 print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(self.train_corpus[sims[index][0]].words)))
# # # #         doc_iter += 1
# # # #
# # # #     print("now let's see if the docs are most similar to themselves: ", collections.Counter(ranks))
# # #
# # #
# # # # remove words from corpus
# # # # words_to_remove_list = ["[deleted]'", '>', "Confirmed", "b'", 'b"', '&amp;#8710;', '&#8710;', '&#916;', '&amp;916;',
# # # #                         '∆', '!delta', 'Δ', '&delta;']
# # #
# # # # dtype = torch.float
# # # # device = torch.device("cpu")
# # #
# # #
# # # # device = torch.device("cpu")
# # # #
# # # # daf = pd.DataFrame([[2,13,8],[45,6,0]])
# # # #
# # # # def df_to_tensor(df):
# # # #     """
# # # #     th
# # #
# # # is method takes a df of values / vectores and returns a tensor of 2 dim/ 3 dim accordingly
# # # #     :return: tensor
# # # #     """
# # # #
# # # #     # get shapes
# # # #     df_rows_num = df.shape[0]
# # # #     df_columns_num = df.shape[1]
# # # #
# # # #     # if values of df is numbers
# # # #     if isinstance(df.iloc[0, 0], numbers.Number):
# # # #         print("new tensor shape is", df_rows_num, ",", df_columns_num)
# # # #         return tr.Tensor(df.values)
# # # #     # if values of df are vectors
# # # #     else:
# # # #         df_value_length = len(df.iloc[0, 0])
# # # #         df_content = df.values
# # # #         tensor = tr.Tensor([[column for column in row] for row in df_content])
# # # #         print("new tensor shape is", df_rows_num, ",", df_columns_num, ",", df_value_length)
# # # #
# # # #         return tensor
# # # #
# # # # daf_t = df_to_tensor(daf)
# # # # print("ok")
# #
# # train = list([3,5,6])
# # dict = dict({'f':5})
# #
# # train_data = train, dict
# #
# # def func2(train, dict):
# #
# #     print(len(train))
# #     print(dict.keys())
# #
# #     return
# #
# # def func(pef, data):
# #
# #     print(pef)
# #     func2(*data)
# #
# #     return
#
# # from sklearn.ensemble import RandomForestClassifier
# # clf = RandomForestClassifier(max_depth=2, random_state=0)
# # clf.fit(df, y)
#
# # print(df)
# # for col in df.columns:
# #     if df[col].dtype == 'int64':
# #         print("bed")
# #
# #
# # func('who', train_data)
#
# import torch as tr
# import pandas as pd
#
#
# def sort_batch(x, length):
#
#     batch_size = x.size(0)                       # get size of batch
#     sorted_length, sorted_idx = length.sort()  # sort the length of sequence samples
#     reverse_idx = tr.linspace(batch_size-1,0,batch_size).long()
#     # reverse_idx = reverse_idx.cuda(GPU_ID)
#
#     sorted_length = sorted_length[reverse_idx]    # for descending order
#     sorted_idx = sorted_idx[reverse_idx]
#     sorted_data = x[sorted_idx]                 # sorted in descending order
#
#     return sorted_data, sorted_length, sorted_idx
#
#
# text = {'col1': [[1, 2], [3, 4], [5, 6]], 'col2': [[7, 8], [9, 1], [1, 5]], 'col3': [[0, 0], [1, 6], [5, 5]],
#         'col4': [[0, 0], [6, 8], [0, 0]]}
# df = pd.DataFrame(data=text)
# df_value_length = len(df.iloc[0, 0])
# df_content = df.values
# tensor = tr.Tensor([[column for column in row] for row in df_content])
# lengths = tr.Tensor([2,4,3])
#
# x = tr.randn(10)
# y, ind = tr.sort(x, 0)
# unsorted = y.new(*y.size())
# unsorted.scatter_(0, ind, y)
# # print((x - unsorted).abs().max())
#
# sort_batch(tensor,lengths)
#
# ###Oshri
# print('** How many people have the word Chief in their job title? (This is pretty tricky) **')
# print(sum(df_salaries.JobTitle.str.contains("(?i)chief")))
# print(sum(df_salaries.JobTitle.str.lower().str.contains("chief")))
#
# df_salaries["Job Title"].str
# def chief_in(title):
#     return 'chief' in title.lower()
#
# print(sum(df_salaries['JobTitle'].apply(lambda x: chief_in(x))))
# print(sum(df_salaries['JobTitle'].apply(lambda x: 'chief' in x.lower())))
# # print(sum(df_salaries['JobTitle'].apply(chief_in)))
#
# print(list(range(3)))

# import torch
# import torch.nn.functional as F
# import joblib
# import os
#
# base_dir = os.path.abspath(os.curdir)
# features_dir = os.path.join(base_dir, "features", "small_data_features")
#
# branch_comments_embedded_text_df_train = joblib.load(os.path.join(features_dir, "branch_comments_embedded_text_df_train.pkl"))
# # max_len = branch_comments_embedded_text_df_train.shape[1]
# # final_tens = tr.Tensor
# # first_row = 1
# # for row in branch_comments_embedded_text_df_train.values:
# #     branch = list()
# #     branch_len = 0
# #     for col in row:
# #         if len(col) > 0:
# #             branch.append(col)
# #             branch_len += 1
# #     tensor = tr.Tensor(branch)
# #     F.pad(tensor, pad=(0, max_len-branch_len), mode='constant', value=0)
# #     if first_row:
# #         first_row = 0
# #         first_tensor = tr.Tensor(branch)
# #     if not first_row:
# #         stacked = tr.stack([first_tensor, tensor], dim=0)
#
# # print(stacked.shape)
#
# branch_submission_dict_train = joblib.load(os.path.join(features_dir, "branch_submission_dict_train.pickle"))
#
# print("check")

import joblib
[accuracy, auc, precision, recall]
measurements_dict = joblib.load('measurements_dict.pkl')
print("bug")