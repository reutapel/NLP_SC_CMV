import gensim
import collections
import smart_open
import time
import sys
import joblib
import pandas as pd
import os


base_directory = os.path.abspath(os.curdir)
data_directory = os.path.join(base_directory, 'data')

# TODO: CHANGE DIRECTORIES TO NEW REPOSITORY AND SIMPLY TO BASE DIRECTORY
# TODO: CHANGE OBJECT TO FIT, TRANSFORM


class Doc2Vec:
    """
    this class creates a model that represent a document with an embedded vector
    """

    def __init__(self, fname, linux, use_file=True, data=None, vector_size=50, min_count=2, epochs=20):
        """
        create a doc2vec model object, corpus, vocabulary and train the model
        :param fname: text file name
        :param bool linux: flag
        :param bool use_file: if using file of data or data frame. if file, fname should be given,
                                else- data frame should be given
        :param data: pandas series with train data - the text to train the model with
        :param vector_size: embedded vector length (number of representative words)
        :param min_count: minimum number of occurrences per word for it to be a representative in the vector
        :param epochs: number of iterations over training corpus
        """

        print(time.asctime(time.localtime(time.time())), ": Class Doc2Vec object created")
        self.train_corpus = list()
        self.test_corpus = list()

        if use_file:
            new_file = self.prepare_raw_text(fname, linux)

            self.train_file = new_file
            self.test_file = new_file
        else:
            self.train_file = None
            self.test_file = None

        self.vocab_len = int()

        self.use_file = use_file
        self.train_data = data

        # instantiate the model
        self.doc2vec_model = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size, min_count=min_count, epochs=epochs)
        print(time.asctime(time.localtime(time.time())), ": Doc2Vec model created")

        self.create_corpus()
        self.build_vocab()
        self.train_doc2vec_model()

    def read_corpus(self, tokens_only=False, is_train=True):
        """
        read the file line-by-line, pre-process each line using a simple gensim pre-processing tool (i.e., tokenize text
         into individual words, remove punctuation, set to lowercase, etc), and return a list of words. Note that,
         for a given file (aka corpus), each continuous line constitutes a single document and the length of each line
         (i.e., document) can vary. Also, to train the model, we'll need to associate a tag/number with each document
         of the training corpus. In our case, the tag is simply the zero-based line number
        :param bool tokens_only: if true, for test with no labels e.g. doc indexes
        :param bool is_train: if using train or test file
        :return:
        """
        if not self.use_file:
            for i, line in self.train_data.iteritems():
                if tokens_only:
                    yield gensim.utils.simple_preprocess(line)
                else:
                    # For training data, add tags
                    yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])

        else:
            if is_train:
                data_file = self.train_file
            else:
                data_file = self.test_file
            with smart_open.smart_open(data_file, encoding="iso-8859-1") as f:
                for i, line in enumerate(f):
                    if tokens_only:
                        yield gensim.utils.simple_preprocess(line)
                    else:
                        # For training data, add tags
                        yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])

        return

    def create_corpus(self):
        """
        method read the files by using read_corpus
        :return:
        """

        print(time.asctime(time.localtime(time.time())), ": starting creating train corpus")
        self.train_corpus = list(self.read_corpus())
        # print(time.asctime(time.localtime(time.time())), ": starting creating test corpus")
        # self.test_corpus = list(self.read_corpus(is_train=False, tokens_only=True))
        # print(time.asctime(time.localtime(time.time())), ": finished creating corpuses")

    def build_vocab(self):
        """
        building vocabulary by the train corpus
        :return:
        """

        print(time.asctime(time.localtime(time.time())), ": starting building vocabulary")
        self.doc2vec_model.build_vocab(self.train_corpus)
        print(time.asctime(time.localtime(time.time())), ": finished building vocabulary")
        self.vocab_len = len(self.doc2vec_model.wv.vocab.keys())
        print(time.asctime(time.localtime(time.time())), ": vocabulary size is:", self.vocab_len)
        # the vocabulary is a dictionary (accessible via model.wv.vocab)

    def train_doc2vec_model(self):
        """
        method that trains the embedding model
        :return:
        """

        print(time.asctime(time.localtime(time.time())), ": begin training model")
        self.doc2vec_model.train(self.train_corpus, total_examples=self.doc2vec_model.corpus_count,
                                 epochs=self.doc2vec_model.epochs)
        print(time.asctime(time.localtime(time.time())), ": finished training model")

    def infer_doc_vector(self, doc):

        """
        :param doc: the document to embed
        :return: the embedded vector
        """
        # print(time.asctime(time.localtime(time.time())), ": infering vector for input: ", doc)
        vec = self.doc2vec_model.infer_vector(doc)
        # print(time.asctime(time.localtime(time.time())), ": vector shape is: ", vec.shape)
        return vec

    def prepare_raw_text(self, file, linux):
        """
        temp method for preparing file of raw text in linux from reut's file, and debug log local/linux
        should be removed
        :param file: file name
        :param linux: flag
        :return: file to run dict on
        """

        if linux:
            comments_with_label = pd.read_csv(file)
            all_submissions_comments_with_label_all_deltalog_final_comment_body = comments_with_label.comment_body
            all_submissions_comments_with_label_all_deltalog_final_comment_body = \
                all_submissions_comments_with_label_all_deltalog_final_comment_body.str.lstrip('b"')
            all_submissions_comments_with_label_all_deltalog_final_comment_body = \
                all_submissions_comments_with_label_all_deltalog_final_comment_body.str.lstrip("b'")

            new_file = "all_submissions_comments_with_label_all_deltalog_final_comment_body.csv"
            all_submissions_comments_with_label_all_deltalog_final_comment_body.to_csv(new_file)
        else:
            new_file = file

        return new_file

    # TODO: create method that cleans text
    # TODO: examine evaluation results

    def evaluate_self_sim(self):
        """
        sanity check: method checks if the model makes a doc most similar to itself
        method prints most, second most, median and least similar vec to a doc
        :return:
        """

        ranks = []
        second_ranks = []
        doc_iter = 0
        for doc_id in range(len(self.train_corpus)):
            inferred_vector = self.doc2vec_model.infer_vector(self.train_corpus[doc_id].words)
            sims = self.doc2vec_model.docvecs.most_similar([inferred_vector], topn=len(self.doc2vec_model.docvecs))
            rank = [docid for docid, sim in sims].index(doc_id)
            ranks.append(rank)
            second_ranks.append(sims[1])
            if doc_iter % 500000 == 0:
                print('Document ({}): «{}»\n'.format(doc_id, ' '.join(self.train_corpus[doc_id].words)))
                print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % self.doc2vec_model)
                for label, index in [('MOST', 0), ('SECOND-MOST', 1),
                                     ('MEDIAN', len(sims) // 2), ('LEAST', len(sims) - 1)]:
                    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(self.train_corpus[sims[index][0]].words)))
            doc_iter += 1

        print("now let's see if the docs are most similar to themselves: ", collections.Counter(ranks))


def main():

    # start logging - redirects print to log file
    old_stdout = sys.stdout
    log_file = open("doc2vec.log", "w")
    sys.stdout = log_file

    # text file
    linux = 0
    if linux:
        csv = "change_my_view/all_submissions_comments_with_label_all_deltalog_final.csv"
    else:
        csv = os.path.join(data_directory, 'comment_body.csv')

    # create object
    doc2vec = Doc2Vec(csv, linux, 50, 2, 30)

    # test embedded vector
    test = "I going to show you a baby. Aalive and health"
    vec_test = doc2vec.infer_doc_vector(test)
    print("vec test is", vec_test)

    # evaluate model performance
    doc2vec.evaluate_self_sim()

    print(time.asctime(time.localtime(time.time())), ": saving doc2vec object ")
    joblib.dump(doc2vec, "doc2vec.pkl")

    # close logging
    sys.stdout = old_stdout
    log_file.close()


if __name__ == '__main__':
    main()
