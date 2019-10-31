import pandas as pd
import os
import joblib
import time
from doc2vec import Doc2Vec
import torch
from pytorch_transformers import *
import datetime
from tqdm import *
from submissions_clusters import *
from bert_model import BertTransformer

class SubmissionsTitleClusters:
    """ creates embedded version of submission titles and clusters them """

    def __init__(self, data, data_directory, embedding_size, take_bert_pooler=True):

        self.data = data
        self.data_directory = data_directory

        # embedding class variables
        self.embedding_size = embedding_size
        self.doc2vec_model = None
        self.doc2vec_fitted_model_file_path = None
        self.take_bert_pooler = take_bert_pooler
        self.poolers_df = None
        self.bert_model = None

    def bert_encoding(self, is_save=False):
        """
        encode all submission titles using BERT model encoder
        :return: fills class variable df bert_encoded_submissions_title_df
        """
        # Load pretrained model/tokenizer
        self.bert_model = BertTransformer(pretrained_weights_shortcut='bert-base-uncased', take_bert_pooler=True)

        # Encode text
        if self.take_bert_pooler:
            # print('starting BERT encoding', datetime.datetime.now())
            poolers_list = list()
            for index, row in tqdm(self.data.iteritems(), total=self.data.shape[0]):
                # pooler = self.bert_model.bert_text_encoding(row, take_bert_pooler=True)
                pooler = self.bert_model.get_text_average_pooler_split_bert(row, max_size=512)
                poolers_list.append(pd.Series(pooler))
            self.poolers_df = pd.DataFrame(poolers_list)
            # print('finished BERT encoding', datetime.datetime.now())
            if is_save:
                print('saving BERT embedded df')
                joblib.dump(self.poolers_df, 'bert_poolers_df.pickle')
        return

    def doc2vec_embedding(self, min_count=2, epochs=200):
        """
        embed using doc2vec
        :return:
        """
        # fit embedding by doc2vec
        print('{}: starting fitting doc2vec model'.format(time.asctime(time.localtime(time.time()))))
        self.doc2vec_model = Doc2Vec(fname='', linux=False, use_file=False, data=self.data,
                                     vector_size=self.embedding_size, min_count=min_count, epochs=epochs)
        print('{}: finish fitting doc2vec model'.format(time.asctime(time.localtime(time.time()))))

        # save trained model
        self.doc2vec_fitted_model_file_path = os.path.join(self.data_directory, 'doc2vec_submission_titles.pkl')
        print('{}: starting saving doc2vec model'.format(time.asctime(time.localtime(time.time()))))
        joblib.dump(self.doc2vec_model, self.doc2vec_fitted_model_file_path)
        print('{}: finishing saving doc2vec model'.format(time.asctime(time.localtime(time.time()))))

        return

    def describe_data(self):

        print('{}: analyzing data '.format(time.asctime(time.localtime(time.time()))))
        data_len = self.data.apply(lambda x: len(x.split(' ')))
        print('describing length of titles statistics: ')
        print(data_len.describe())

        data_splitted = self.data.apply(lambda x: x.split(' '))
        corpus = list(set([a for b in data_splitted.tolist() for a in b]))
        print('unique words count is: ', len(corpus))


def main():

    # define paths
    base_directory = os.path.abspath(os.curdir)
    data_directory = os.path.join(base_directory, 'features_to_use')

    # load data
    data_file_path = os.path.join(data_directory, 'all_train_data.csv')
    all_train_data = pd.read_csv(data_file_path)
    print("raw data shape: ", all_train_data.shape)

    # take only submission titles
    all_train_data_submission_title = all_train_data['submission_title'].copy()
    all_train_data_submission_title_unique = all_train_data_submission_title.unique()
    print("unique data shape: ", all_train_data_submission_title_unique.shape)
    del all_train_data

    # remove CMV prefix from titles that have it
    all_train_data_submission_title_unique = \
        pd.Series(all_train_data_submission_title_unique).apply(lambda x: x[5:] if x.startswith('CMV:') else x)

    # create class obj
    sub_title_cluster_obj = SubmissionsTitleClusters(data=all_train_data_submission_title_unique,
                                                     embedding_size=300,
                                                     data_directory=data_directory, take_bert_pooler=True)
    sub_title_cluster_obj.describe_data()

    # encode
    # doc2vec
    # sub_title_cluster_obj.doc2vec_embedding(min_count=2, epochs=1000)
    # evaluate embedding
    # define number of doc to test doc2vec embedding
    # num_of_docs_to_test = 5
    # sub_title_cluster_obj.doc2vec_model.evaluate_self_sim(num_of_docs_to_test)

    sub_title_cluster_obj.bert_encoding(is_save=True)
    # for debugging
    # poolers_df = joblib.load('bert_poolers_df.pickle')
    # cluster embedded data
    submissions_clusters_obj = SubmissionsClusters(sub_title_cluster_obj.poolers_df)
    # reduce dimensions in 3 methods, tsne with pca, tsne without pca, umap
    submission_title_bert_embedded_x_tsne_with_pca = submissions_clusters_obj.tsne_dim_reduce(plot_tsne_results=True,
                                                                                  is_pca_pre=True, pca_dim_pre=50,
                                                                                  n_components=2, perplexity=30.0,
                                                                                  early_exaggeration=4.0, random_state=0
                                                                                  , method='barnes_hut', angle=0.5,
                                                                                  learning_rate=1000, n_iter=1000,
                                                                                  n_iter_without_progress=30,
                                                                                  metric='euclidean', init='pca',
                                                                                  verbose=0)

    submission_title_bert_embedded_x_tsne = submissions_clusters_obj.tsne_dim_reduce(plot_tsne_results=True,
                                                                                  is_pca_pre=False, pca_dim_pre=50,
                                                                                  n_components=2, perplexity=30.0,
                                                                                  early_exaggeration=4.0, random_state=0
                                                                                  , method='barnes_hut', angle=0.5,
                                                                                  learning_rate=1000, n_iter=1000,
                                                                                  n_iter_without_progress=30,
                                                                                  metric='euclidean', init='pca',
                                                                                  verbose=0)

    submission_title_bert_embedded_x_umap = submissions_clusters_obj.umap_dim_reduce(n_neighbors=30, min_dist=0.0,
                                                                                     n_components=2, random_state=42)

    joblib.dump(submission_title_bert_embedded_x_tsne, 'submission_title_bert_embedded_x_tsne.pickle')
    joblib.dump(submission_title_bert_embedded_x_tsne_with_pca, 'submission_title_bert_embedded_x_tsne_with_pca.pickle')
    joblib.dump(submission_title_bert_embedded_x_umap, 'submission_title_bert_embedded_x_umap.pickle')

    # cluster in
    # GMM clustering on different embeddings
    submission_title_bert_embedded_poolers_y_gmm = submissions_clusters_obj.gmm_cluster(
        sub_title_cluster_obj.poolers_df, n_components=10, covariance_type='full')
    submission_title_bert_embedded_tsne_with_pca_y_gmm = submissions_clusters_obj.gmm_cluster(
        submission_title_bert_embedded_x_tsne_with_pca, n_components=10, covariance_type='full')
    submission_title_bert_embedded_tsne_y_gmm = submissions_clusters_obj.gmm_cluster(
        submission_title_bert_embedded_x_tsne, n_components=10, covariance_type='full')
    submission_title_bert_embedded_umap_y_gmm = submissions_clusters_obj.gmm_cluster(
        submission_title_bert_embedded_x_umap, n_components=10, covariance_type='full')

    joblib.dump(submission_title_bert_embedded_poolers_y_gmm, 'submission_title_bert_embedded_poolers_y_gmm.pickle')
    joblib.dump(submission_title_bert_embedded_tsne_with_pca_y_gmm, 'submission_title_bert_embedded_tsne_with_pca_y_gmm.pickle')
    joblib.dump(submission_title_bert_embedded_tsne_y_gmm, 'submission_title_bert_embedded_tsne_y_gmm.pickle')
    joblib.dump(submission_title_bert_embedded_x_umap, 'submission_title_bert_embedded_x_umap.pickle')

    # HDBSCAN clustering on different embeddings
    submission_title_bert_embedded_poolers_y_hdbscan = submissions_clusters_obj.hdbscan_cluster(
        sub_title_cluster_obj.poolers_df)
    submission_title_bert_embedded_tsne_with_pca_y_hdbscan = submissions_clusters_obj.hdbscan_cluster(
        submission_title_bert_embedded_x_tsne_with_pca)
    submission_title_bert_embedded_tsne_y_hdbscan = submissions_clusters_obj.hdbscan_cluster(
        submission_title_bert_embedded_x_tsne)
    submission_title_bert_embedded_umap_y_hdbscan = submissions_clusters_obj.hdbscan_cluster(
        submission_title_bert_embedded_x_umap)
    # todo: try also OPTICS algorithm for clustering:
    #  https://scikit-learn.org/stable/auto_examples/cluster/plot_optics.html#sphx-glr-auto-examples-cluster-plot-optics-py
    joblib.dump(submission_title_bert_embedded_poolers_y_hdbscan, 'submission_title_bert_embedded_poolers_y_hdbscan.pickle')
    joblib.dump(submission_title_bert_embedded_tsne_with_pca_y_hdbscan, 'submission_title_bert_embedded_tsne_with_pca_y_hdbscan.pickle')
    joblib.dump(submission_title_bert_embedded_tsne_y_hdbscan, 'submission_title_bert_embedded_tsne_y_hdbscan.pickle')
    joblib.dump(submission_title_bert_embedded_umap_y_hdbscan, 'submission_title_bert_embedded_umap_y_hdbscan.pickle')

    print('evaluating embedding and clustering methods:')
    print('silhouette_score- NOTE boundeed between -1 to 1, higher is better, 0 is overlapping clusters, biased for convex ' 
    'clusters and not density based like DBSCAN'
    ' calinski_harabasz_score-NOTE higher is better, biased for convex clusters and not density based like DBSCAN '
    'davies_bouldin_score -NOTE average Euclidean distance within cluster/ between centroids ratio within the closer to' 
    ' 0 the better, biased for convex clusters and not density based like DBSCAN, and limited to Euclidean only')
    # evaluate clustering
    # GMM
    submissions_clusters_obj.evaluate_clusters(data=sub_title_cluster_obj.poolers_df,
                                               cluster_labels=submission_title_bert_embedded_poolers_y_gmm,
                                               cluster_method_name='GMM WITH BERT POOLERS EMBEDDING')

    submissions_clusters_obj.evaluate_clusters(data=submission_title_bert_embedded_x_tsne_with_pca,
                                               cluster_labels=submission_title_bert_embedded_tsne_with_pca_y_gmm,
                                               cluster_method_name='GMM WITH TSNE WITH PCA')

    submissions_clusters_obj.evaluate_clusters(data=submission_title_bert_embedded_x_tsne,
                                               cluster_labels=submission_title_bert_embedded_tsne_y_gmm,
                                               cluster_method_name='GMM WITH TSNE')

    submissions_clusters_obj.evaluate_clusters(data=submission_title_bert_embedded_x_umap,
                                               cluster_labels=submission_title_bert_embedded_umap_y_gmm,
                                               cluster_method_name='GMM WITH UMAP')
    # HDBSCAN
    submissions_clusters_obj.evaluate_clusters(data=sub_title_cluster_obj.poolers_df,
                                               cluster_labels=submission_title_bert_embedded_poolers_y_hdbscan,
                                               cluster_method_name='HDBSCAN WITH BERT POOLERS EMBEDDING')

    submissions_clusters_obj.evaluate_clusters(data=submission_title_bert_embedded_x_tsne_with_pca,
                                               cluster_labels=submission_title_bert_embedded_tsne_with_pca_y_hdbscan,
                                               cluster_method_name='HDBSCAN WITH TSNE WITH PCA')

    submissions_clusters_obj.evaluate_clusters(data=submission_title_bert_embedded_x_tsne,
                                               cluster_labels=submission_title_bert_embedded_tsne_y_hdbscan,
                                               cluster_method_name='HDBSCAN WITH TSNE')

    submissions_clusters_obj.evaluate_clusters(data=submission_title_bert_embedded_x_umap,
                                               cluster_labels=submission_title_bert_embedded_umap_y_hdbscan,
                                               cluster_method_name='HDBSCAN WITH UMAP')



if __name__ == '__main__':
    main()

