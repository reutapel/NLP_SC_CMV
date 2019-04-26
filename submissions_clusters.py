import numpy as np
import pandas as pd
import os
import joblib
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import time


base_directory = os.path.abspath(os.curdir)
# base_directory = os.path.join(base_directory, 'to_server')
data_directory = os.path.join(base_directory, 'data')
save_data_directory = os.path.join(data_directory, 'filter_submissions')
trained_models_directory = os.path.join(base_directory, 'trained_models')


class SubmissionsClusters:
    def __init__(self):
        print(f'{time.asctime(time.localtime(time.time()))}: start loading data and doc2vec model')
        self.branch_comments_info_df = pd.read_csv(
            os.path.join(save_data_directory, 'comments_label_branch_info_after_remove.csv'))
        doc2vec_dir = os.path.join(trained_models_directory, 'new', 'doc2vec_model.pkl')
        self.doc2vec_model = joblib.load(doc2vec_dir)
        self.embedded_submission_text = pd.DataFrame()
        print(f'{time.asctime(time.localtime(time.time()))}: finish loading data and doc2vec model')

    def create_embedded(self):
        print(f'{time.asctime(time.localtime(time.time()))}: start creating data')
        all_submissions = self.branch_comments_info_df[['submission_body', 'submission_id']]
        all_submissions = all_submissions.drop_duplicates()
        for index, row in all_submissions.iterrows():
            vector = pd.DataFrame(
                data=[[row['submission_id'], row['submission_body'],
                      self.doc2vec_model.infer_doc_vector(row['submission_body'])]],
                columns=['submission_id', 'submission_body', 'submission_embedded_body'])
            self.embedded_submission_text = self.embedded_submission_text.append(vector, ignore_index=True)

        self.embedded_submission_text.to_csv(os.path.join(trained_models_directory, 'new', 'embedded_submission_text'))
        print(f'{time.asctime(time.localtime(time.time()))}: finish creating data')

    def tsne_cluster(self):
        # perplexity - denotes the effective number of neighbors every example has range[5-50],
        # early_exaggeration - how spaced you want the clusters to be
        # learning rate - kl-divergance is non convex, with gradient descent no guarantee for non local minima
        print(f'{time.asctime(time.localtime(time.time()))}: start TSNE cluster')
        pca_model = PCA(n_components=50)
        pca_vector = pca_model.fit_transform(self.embedded_submission_text['submission_embedded_body'])
        tnse_model = TSNE(n_components=2, perplexity=30.0, early_exaggeration=4.0, random_state=0, method='barnes_hut',
                          angle=0.5, learning_rate=1000, n_iter=1000, n_iter_without_progress=30, metric='euclidean',
                          init='pca', verbose=0)
        x_tsne = tnse_model.fit_transform(pca_vector)

        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.scatter(x_tsne[:, 0], x_tsne[:, 1])

        plt.show()
        print(f'{time.asctime(time.localtime(time.time()))}: finish TSNE cluster')

        return

    def gmm_cluster(self):
        print(f'{time.asctime(time.localtime(time.time()))}: start GMM cluster')
        gmm = GaussianMixture(n_components=10, covariance_type='full')
        gmm.fit(self.embedded_submission_text['submission_embedded_body'])
        y_pred = gmm.predict(self.embedded_submission_text['submission_embedded_body'])
        final_gmm_result =\
            pd.concat([self.embedded_submission_text[['submission_body', 'submission_id']], y_pred], axis=1)
        final_gmm_result.to_csv(os.path.join(trained_models_directory, 'new', 'gmm_results'))

        print(f'{time.asctime(time.localtime(time.time()))}: finish GMM cluster')


def main():
    submissions_clusters_obj = SubmissionsClusters()
    submissions_clusters_obj.create_embedded()
    submissions_clusters_obj.tsne_cluster()
    submissions_clusters_obj.gmm_cluster()


if __name__ == '__main__':
    main()
