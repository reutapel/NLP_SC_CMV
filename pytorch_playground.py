# # import torch as tr
# # import pandas as pd
# # import numpy as np
# # import torch
# # import numbers
# # import torch.autograd as autograd
# # import torch.nn as nn
# # import torch.nn.functional as F
# # import torch.optim as optim
# # #
# # # ################################################ TENSOR GRADIENTS #####################################################
# # # ######################################################################################################################
# # x = torch.randn(2, 2)
# # y = torch.randn(2, 2)
# # # By default, user created Tensors have ``requires_grad=False``
# # print(x.requires_grad, y.requires_grad)
# # z = x + y
# # # So you can't backprop through z
# # print(z.grad_fn)
# #
# # # ``.requires_grad_( ... )`` changes an existing Tensor's ``requires_grad``
# # # flag in-place. The input flag defaults to ``True`` if not given.
# # x = x.requires_grad_()
# # y = y.requires_grad_()
# # # z contains enough information to compute gradients, as we saw above
# # z = x + y
# # print(z.grad_fn)
# # # If any input to an operation has ``requires_grad=True``, so will the output
# # print(z.requires_grad)
# #
# # # Now z has the computation history that relates itself to x and y
# # # Can we just take its values, and **detach** it from its history?
# # new_z = z.detach()
# #
# # # ... does new_z have information to backprop to x and y?
# # # NO!
# # print(new_z.grad_fn)
# # # And how could it? ``z.detach()`` returns a tensor that shares the same storage
# # # as ``z``, but with the computation history forgotten. It doesn't know anything
# # # about how it was computed.
# # # In essence, we have broken the Tensor away from its past history
# #
# # # You can also stop autograd from tracking history on Tensors with .requires_grad``=True by wrapping the code block in
# # # ``with torch.no_grad():
# #
# # print(x.requires_grad)
# # print((x ** 2).requires_grad)
# #
# # with torch.no_grad():
# #     print((x ** 2).requires_grad)
# #
# # ######################################################################################################################
# # ######################################################################################################################
# #
# #
# # ############## RELU ###############
# #
# # data = torch.randn(2, 2)
# # print(data)
# # print(F.relu(data))
# #
# # ###################################a
# #
# #
# # ############## SOFTMAX ###############
# # # just a non-linearity, but it is special in that it usually is the last operation done in a network.
# # # This is because it takes in a vector of real numbers and returns a probability distribution
# # # Softmax is also in torch.nn.functional
# # data = torch.randn(5)
# # print(data)
# # print(F.softmax(data, dim=0))
# # print(F.softmax(data, dim=0).sum())  # Sums to 1 because it is a distribution!
# # print(F.log_softmax(data, dim=0))  # theres also log_softmax
# #
# # ####################################
# #
# #
# #
# # T_data = [[[1., 2.], [3., 4.]],
# #           [[5., 6.], [7., 8.]]]
# # T = torch.tensor(T_data, requires_grad=True)
# # print(T[0,1,1].item())
# # print(T.grad_fn)
# #
# # # Tensor factory methods have a ``requires_grad`` flag
# # x = torch.tensor([1., 2., 3], requires_grad=True)
# #
# # # With requires_grad=True, you can still do all the operations you previously
# # # could
# # y = torch.tensor([4., 5., 6], requires_grad=True)
# # z = x + y
# # print(z)
# #
# # # BUT z knows something extra.
# # print(z.grad_fn)
# #
# # # tr.manual_seed(1)
# # # a = tr.rand(3,4,5)
# # #
# # # print(tr.__version__)
# #
# # # rnn = tr.nn.LSTMCell(10, 20)
# # # print("rnn is", rnn)
# # # input = tr.randn(6, 3, 10)
# # # print("input is", input)
# # # hx = tr.randn(3, 20)
# # # print("hx is", hx)
# # # cx = tr.randn(3, 20)
# # # print("cx is", cx)
# # # output = []
# # # for i in range(6):
# # #     hx, cx = rnn(input[i], (hx, cx))
# # #     print("hx is", hx)
# # #     print("cx is", cx)
# # #     output.append(hx)
# # # #     print("output is", output)
# # #
# # # raw_text_df = pd.Series([['sdfsdfsd'],['sdfsg']])
# # # raw_text_tensor = tr.Tensor(raw_text_df.values)
# # # print(raw_text_tensor)
# #
# #
# # # lstm = nn.LSTM(3, 3)  # Input dim is 3, output dim is 3
# # # inputs = [torch.randn(1, 3) for _ in range(5)]  # make a sequence of length 5
# # # print(inputs)
# #
# # # csv = "all_submissions_comments_with_label_all_deltalog_final.csv"
# # # comments_with_label = pd.read_csv(csv)
# # # all_submissions_comments_with_label_all_deltalog_final_comment_body = comments_with_label.comment_body
# # # all_submissions_comments_with_label_all_deltalog_final_comment_body = \
# # #     all_submissions_comments_with_label_all_deltalog_final_comment_body.str.lstrip('b"')
# # # all_submissions_comments_with_label_all_deltalog_final_comment_body = \
# # #     all_submissions_comments_with_label_all_deltalog_final_comment_body.str.lstrip("b'")
# # # # all_submissions_comments_with_label_all_deltalog_final_comment_body = \
# # # #     all_submissions_comments_with_label_all_deltalog_final_comment_body[all_submissions_comments_with_label_all_deltalog_final_comment_body.str.contains("Confirmed") == False]
# # # # all_submissions_comments_with_label_all_deltalog_final_comment_body = \
# # # #     all_submissions_comments_with_label_all_deltalog_final_comment_body[all_submissions_comments_with_label_all_deltalog_final_comment_body.str.contains("[deleted]") == False]
# # # all_submissions_comments_with_label_all_deltalog_final_comment_body.to_csv("all_submissions_comments_with_label_all_deltalog_final_comment_body.csv")
# # # print("Eli")
# #
# # # def evaluate_self_sim(self):
# # #     """
# # #     sanity check: method checks if the model makes a doc most similar to itself
# # #     method prints most, second most, median and least similar vec to a doc
# # #     :return:
# # #     """
# # #
# # #     ranks = []
# # #     second_ranks = []
# # #     doc_iter = 0
# # #     for doc_id in range(len(self.train_corpus)):
# # #         inferred_vector = self.doc2vec_model.infer_vector(self.train_corpus[doc_id].words)
# # #         sims = self.doc2vec_model.docvecs.most_similar([inferred_vector], topn=len(self.doc2vec_model.docvecs))
# # #         rank = [docid for docid, sim in sims].index(doc_id)
# # #         ranks.append(rank)
# # #         second_ranks.append(sims[1])
# # #         if doc_iter % 500000 == 0:
# # #             print('Document ({}): «{}»\n'.format(doc_id, ' '.join(self.train_corpus[doc_id].words)))
# # #             print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % self.doc2vec_model)
# # #             for label, index in [('MOST', 0), ('SECOND-MOST', 1),
# # #                                  ('MEDIAN', len(sims) // 2), ('LEAST', len(sims) - 1)]:
# # #                 print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(self.train_corpus[sims[index][0]].words)))
# # #         doc_iter += 1
# # #
# # #     print("now let's see if the docs are most similar to themselves: ", collections.Counter(ranks))
# #
# #
# # # remove words from corpus
# # # words_to_remove_list = ["[deleted]'", '>', "Confirmed", "b'", 'b"', '&amp;#8710;', '&#8710;', '&#916;', '&amp;916;',
# # #                         '∆', '!delta', 'Δ', '&delta;']
# #
# # # dtype = torch.float
# # # device = torch.device("cpu")
# #
# #
# # # device = torch.device("cpu")
# # #
# # # daf = pd.DataFrame([[2,13,8],[45,6,0]])
# # #
# # # def df_to_tensor(df):
# # #     """
# # #     th
# #
# # is method takes a df of values / vectores and returns a tensor of 2 dim/ 3 dim accordingly
# # #     :return: tensor
# # #     """
# # #
# # #     # get shapes
# # #     df_rows_num = df.shape[0]
# # #     df_columns_num = df.shape[1]
# # #
# # #     # if values of df is numbers
# # #     if isinstance(df.iloc[0, 0], numbers.Number):
# # #         print("new tensor shape is", df_rows_num, ",", df_columns_num)
# # #         return tr.Tensor(df.values)
# # #     # if values of df are vectors
# # #     else:
# # #         df_value_length = len(df.iloc[0, 0])
# # #         df_content = df.values
# # #         tensor = tr.Tensor([[column for column in row] for row in df_content])
# # #         print("new tensor shape is", df_rows_num, ",", df_columns_num, ",", df_value_length)
# # #
# # #         return tensor
# # #
# # # daf_t = df_to_tensor(daf)
# # # print("ok")
#
# train = list([3,5,6])
# dict = dict({'f':5})
#
# train_data = train, dict
#
# def func2(train, dict):
#
#     print(len(train))
#     print(dict.keys())
#
#     return
#
# def func(pef, data):
#
#     print(pef)
#     func2(*data)
#
#     return
import pandas as pd

df = pd.DataFrame({'col1':[1,2,3,4,5], 'col2':list('abcab'),  'col3':list('ababb')})
# cat_columns = df.select_dtypes(['category']).columns
# df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
# print(df)
y=list([1, 1, 0, 1, 0])

for col_name in list(set(df.columns).difference(set("injury_savirity"))):
    if(df[col_name].dtype == 'object'):
        df[col_name]= df[col_name].astype('category')
        df[col_name] = df[col_name].cat.codes

# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(max_depth=2, random_state=0)
# clf.fit(df, y)
print(df[~'col1'])

# print(df)
# for col in df.columns:
#     if df[col].dtype == 'int64':
#         print("bed")
#
#
# func('who', train_data)

