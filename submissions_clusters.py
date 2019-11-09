import numpy as np
import pandas as pd
import os
import joblib
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import time
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim.sklearn_api import ldamodel
from sklearn import metrics
import umap
import hdbscan
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import nltk


base_directory = os.path.abspath(os.curdir)
# base_directory = os.path.join(base_directory, 'to_server')
data_directory = os.path.join(base_directory, 'data')
save_data_directory = os.path.join(data_directory, 'filter_submissions')
trained_models_directory = os.path.join(base_directory, 'trained_models')


class SubmissionsClusters:
    def __init__(self, data, number_of_topics=15, use_topic=False, hdbscan_min_cluster_size=5):
        # self.number_of_topics = number_of_topics
        # embedded_file_path = os.path.join(os.path.join(trained_models_directory, 'new', 'embedded_submission_text.pkl'))
        # if not os.path.isfile(embedded_file_path):
        #     print(f'{time.asctime(time.localtime(time.time()))}: '
        #           f'start loading data and doc2vec model and create text representation')
        #     self.branch_comments_info_df = pd.read_csv(
        #         os.path.join(save_data_directory, 'comments_label_branch_info_after_remove.csv'))
        #     doc2vec_dir = os.path.join(trained_models_directory, 'new', 'doc2vec_model.pkl')
        #     self.doc2vec_model = joblib.load(doc2vec_dir)
        #     self.embedded_submission_text = pd.DataFrame()
        #     if not use_topic:  # if using embedded as text representations
        #         self.create_embedded()
        #         self.text_presentation = self.embedded_submission_text
        #     else:  # if needs to create topics and we don't have embedded yet - use the orig data
        #         all_submissions = self.branch_comments_info_df[['submission_body', 'submission_id']]
        #         all_submissions = all_submissions.drop_duplicates()
        #         self.embedded_submission_text = all_submissions
        #         self.text_presentation = self.topic_model()
        #     print(f'{time.asctime(time.localtime(time.time()))}: '
        #           f'finish loading data and doc2vec model and create text representation')
        # else:
        #     self.embedded_submission_text = pd.read_pickle(embedded_file_path)
        #     if use_topic:
        #         self.text_presentation = self.topic_model()
        #     else:
        #         self.text_presentation = self.embedded_submission_text
        self.data = data
        self.number_of_topics = number_of_topics
        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size

    def create_embedded(self):
        print(f'{time.asctime(time.localtime(time.time()))}: start creating embedded data')
        all_submissions = self.branch_comments_info_df[['submission_body', 'submission_id']]
        all_submissions = all_submissions.drop_duplicates()
        for index, row in all_submissions.iterrows():
            vector = pd.DataFrame(
                data=[[row['submission_id'], row['submission_body'],
                      self.doc2vec_model.infer_doc_vector(row['submission_body'])]],
                columns=['submission_id', 'submission_body', 'submission_embedded_body'])
            self.embedded_submission_text = self.embedded_submission_text.append(vector, ignore_index=True)

        self.embedded_submission_text.to_pickle(os.path.join(trained_models_directory, 'new',
                                                             'embedded_submission_text.pkl'))
        print(f'{time.asctime(time.localtime(time.time()))}: finish creating embedded data')

    def umap_dim_reduce(self, n_neighbors=30, min_dist=0.0, n_components=2, random_state=42):

        x_umap = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components,
                           random_state=random_state).fit_transform(self.data)
        return x_umap

    def tsne_dim_reduce(self, plot_tsne_results=True, is_pca_pre=True, pca_dim_pre=50, n_components=2, perplexity=30.0,
                     early_exaggeration=4.0, random_state=0, method='barnes_hut', angle=0.5, learning_rate=1000,
                     n_iter=1000, n_iter_without_progress=30, metric='euclidean', init='pca', verbose=0):

        # perplexity - denotes the effective number of neighbors every example has range[5-50],
        # early_exaggeration - how spaced you want the clusters to be
        # learning rate - kl-divergance is non convex, with gradient descent no guarantee for non local minima
        print(f'{time.asctime(time.localtime(time.time()))}: start TSNE dimensionality reduction')

        # declare TSNE model
        tnse_model = TSNE(n_components=n_components, perplexity=perplexity, early_exaggeration=early_exaggeration,
                          random_state=random_state, method=method, angle=angle, learning_rate=learning_rate,
                          n_iter=n_iter, n_iter_without_progress=n_iter_without_progress, metric=metric,
                          init=init, verbose=verbose)

        if is_pca_pre:
            print('reducing to dim 50 with PCA')
            pca_model = PCA(n_components=pca_dim_pre)
            pca_vector = pca_model.fit_transform(self.data)
            x_tsne = tnse_model.fit_transform(pca_vector)
        else:
            x_tsne = tnse_model.fit_transform(self.data)

        # plot clustering
        if plot_tsne_results:
            plt.figure(figsize=(10, 5))
            plt.subplot(121)
            plt.scatter(x_tsne[:, 0], x_tsne[:, 1])
            plt.show()

        print(f'{time.asctime(time.localtime(time.time()))}: finish TSNE dimensionality reduction')

        return x_tsne

    def gmm_cluster(self, data, n_components=10, covariance_type='full'):

        print(f'{time.asctime(time.localtime(time.time()))}: start GMM cluster')

        gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type)
        gmm.fit(data)

        cluster_labels = gmm.predict(data)

        # final_gmm_result =\
        #     pd.concat([self.embedded_submission_text[['submission_body', 'submission_id']], pd.Series(y_pred)], axis=1)
        # final_gmm_result.to_csv(os.path.join(trained_models_directory, 'new', 'gmm_results.csv'))

        print(f'{time.asctime(time.localtime(time.time()))}: finish GMM cluster')

        return cluster_labels

    def hdbscan_cluster(self, data):
        cluster_labels = hdbscan.HDBSCAN(self.hdbscan_min_cluster_size).fit_predict(data)
        return cluster_labels

    def topic_model(self):
        """
        Calculate the topic model for all the units, the probability that the comment has each of the topics
        :return: pandas DF[number_of_units, number_of_topics] - the probability for each comment and topic
        """
        print('{}: Start topic model'.format((time.asctime(time.localtime(time.time())))))
        # Clean the data
        print('{}: Clean the data'.format((time.asctime(time.localtime(time.time())))))

        data_clean = {row['submission_id']: clean(row['submission_body']).split()
                      for index, row in self.embedded_submission_text.iterrows()}

        # Creating the term dictionary of our corpus, where every unique term is assigned an index.
        print('{}: Create the dictionary'.format(time.asctime(time.localtime(time.time()))))
        dictionary = gensim.corpora.Dictionary(data_clean.values())

        # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
        print('{}: Create data term matrix'.format(time.asctime(time.localtime(time.time()))))
        data_term_matrix = {index: dictionary.doc2bow(doc) for index, doc in data_clean.items()}

        # Get topics for the data
        print('{}: Predict topics'.format(time.asctime(time.localtime(time.time()))))

        lda_model = ldamodel.LdaTransformer(num_topics=self.number_of_topics, id2word=dictionary, passes=50,
                                            minimum_probability=0)
        result = lda_model.transform(list(data_term_matrix.values()))

        print('{}: Create final topic model'.format(time.asctime(time.localtime(time.time()))))
        comment_ids_df = pd.DataFrame(list(data_term_matrix.keys()), columns=['submission_id'])
        result_columns = [i for i in range(self.number_of_topics)]
        topic_model_result_df = pd.DataFrame(result, columns=result_columns)

        print('{}: Save final topic model'.format(time.asctime(time.localtime(time.time()))))
        topic_model_final_result = pd.concat([comment_ids_df, topic_model_result_df], axis=1)
        print('{}: Finish topic model'.format((time.asctime(time.localtime(time.time())))))

        return topic_model_final_result

    def evaluate_clusters(self, data, cluster_labels, cluster_method_name) -> dict:
        # metrics.pairwise.pairwise_distances
        print('evaluating ', cluster_method_name)
        silhouette_score_euc = metrics.silhouette_score(data, cluster_labels, metric='euclidean')
        print('silhouette_score euclidean is: ', str(silhouette_score_euc))
        silhouette_score_cos = metrics.silhouette_score(data, cluster_labels, metric='cosine')
        print('silhouette_score cosine is: ', str(silhouette_score_cos))

        calinski_harabasz_score = metrics.calinski_harabasz_score(data, cluster_labels)
        print('calinski_harabasz_score (aka variance-ratio) is: ', str(calinski_harabasz_score))

        davies_bouldin_score = metrics.davies_bouldin_score(data, cluster_labels)
        print('davies_bouldin_score is: ', str(davies_bouldin_score))

        return {'cluster_method_name': cluster_method_name,
                'silhouette_score_euc': silhouette_score_euc,
                'silhouette_score_cos': silhouette_score_cos,
                'calinski_harabasz_score' :calinski_harabasz_score,
                'davies_bouldin_score': davies_bouldin_score}


def get_cluster_top_words(df, method_name, cluster_num, top_n=10, print_freq_df=True, directory=base_directory):

    freq_df = df.str.split(expand=True).stack().value_counts()

    stopwords = nltk.corpus.stopwords.words('english')
    freq_df = freq_df[~freq_df.index.isin(stopwords + ['CMV:', 'CMV', 'CMV.'])]

    if print_freq_df:
        print(freq_df.head(top_n))

    # print graph
    matplotlib.style.use('ggplot')
    freq_df.head(top_n).plot.bar(rot=0)

    freq_df.to_csv(os.path.join(directory, f'top_words_{method_name}_cluster_{str(cluster_num)}.csv'))
    return


def clean(text):
    """
    This function clean a text from stop words and punctuations and them lemmatize the words
    :param str text: the text we want to clean
    :return: str normalized: the cleaned text
    """

    # for topic modeling clean text
    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    text = text.lstrip('b').strip('"').strip("'").strip(">")
    stop_free = " ".join([i for i in text.lower().split() if i not in stop])
    punc_free = "".join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized


def main():
    number_of_topics = 15
    submissions_clusters_obj = SubmissionsClusters(number_of_topics)

    submissions_clusters_obj.topic_model()
    submissions_clusters_obj.tsne_cluster()
    submissions_clusters_obj.gmm_cluster()


if __name__ == '__main__':
    main()
