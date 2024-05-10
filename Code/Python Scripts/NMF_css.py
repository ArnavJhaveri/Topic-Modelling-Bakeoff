"""
1. Dataset passed in must be a CSV file that can be opened using pd.read_csv()
2. perc determines how much of the dataset is used
3. num_topics_word is set to 500 top words
4. num_topics_list is set to [100] but can be changed or appended to, to get more results
5. text_column, label_column, and id_column must all be set according to your database
  5.1 text_column is the actual text
  5.2 label_column is what you regress over / predict
  5.3 id_column is the unique identifier for each message
6. dataset_path must be set to where the dataset is stored
7. the name of the folder (be sure to include '/' at the end) -- usually the name of the dataset (i.e. twitter, amazon, etc.)
8. logit should be set to True if target variable is discrete and False if continous
"""

import os
import errno
import nltk
import pandas as pd
from nltk.corpus import stopwords as stop_words
import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import string
from gensim.utils import deaccent
import warnings
import scipy.sparse
from sklearn.preprocessing import OneHotEncoder
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import KeyedVectors
import gensim.downloader as api
from scipy.spatial.distance import cosine
import abc
import re
import itertools
import pickle

is_multi = False
is_combined = not is_multi

nltk.download('stopwords')

language = 'en'
stopwords = list(stop_words.words('english'))

num_topics_word = 500
num_topics_list = [1000]

# change these accordingly
text_column = 'message'
label_column = 'age'
id_column = 'Unnamed: 0'
dataset_path = 'data/twitter.csv'
folder = 'twitter/'

dataset = pd.read_csv(dataset_path)[[id_column, text_column, label_column]].rename(columns = {id_column: 'message_id', text_column: 'message', label_column: 'label'}).dropna()

# shuffles to avoid biases in data
arr = sklearn.utils.shuffle(np.arange(len(dataset)), random_state=42)
dataset = dataset.iloc[arr].reset_index(drop=True)

train, test, message, test_message = train_test_split(dataset, dataset['message'], test_size = 0.20, random_state = 42)

train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)
message.reset_index(drop=True, inplace=True)
test_message.reset_index(drop=True, inplace=True)

message = list(message)
test_message = list(test_message)

class WhiteSpacePreprocessingStopwords():
    """
    Provides a very simple preprocessing script that filters infrequent tokens from text
    """

    def __init__(self, documents, stopwords_list=None, vocabulary_size=2000, max_df=1.0, min_words=1,
                 remove_numbers=True):
        """

        :param documents: list of strings
        :param stopwords_list: list of the stopwords to remove
        :param vocabulary_size: the number of most frequent words to include in the documents. Infrequent words will be discarded from the list of preprocessed documents
        :param max_df : float or int, default=1.0
        When building the vocabulary ignore terms that have a document
        frequency strictly higher than the given threshold (corpus-specific
        stop words).
        If float in range [0.0, 1.0], the parameter represents a proportion of
        documents, integer absolute counts.
        This parameter is ignored if vocabulary is not None.
        :param min_words: int, default=1. Documents with less words than the parameter
        will be removed
        :param remove_numbers: bool, default=True. If true, numbers are removed from docs
        """
        self.documents = documents
        if stopwords_list is not None:
            self.stopwords = set(stopwords_list)
        else:
            self.stopwords = []

        self.vocabulary_size = vocabulary_size
        self.max_df = max_df
        self.min_words = min_words
        self.remove_numbers = remove_numbers

    def preprocess(self):
        """
        Note that if after filtering some documents do not contain words we remove them. That is why we return also the
        list of unpreprocessed documents.

        :return: preprocessed documents, unpreprocessed documents and the vocabulary list
        """
        preprocessed_docs_tmp = self.documents
        preprocessed_docs_tmp = [deaccent(doc.lower()) for doc in preprocessed_docs_tmp]
        preprocessed_docs_tmp = [doc.translate(
            str.maketrans(string.punctuation, ' ' * len(string.punctuation))) for doc in preprocessed_docs_tmp]
        if self.remove_numbers:
            preprocessed_docs_tmp = [doc.translate(str.maketrans("0123456789", ' ' * len("0123456789")))
                                     for doc in preprocessed_docs_tmp]
        preprocessed_docs_tmp = [' '.join([w for w in doc.split() if len(w) > 0 and w not in self.stopwords])
                                 for doc in preprocessed_docs_tmp]

        vectorizer = CountVectorizer(max_features=self.vocabulary_size, max_df=self.max_df)
        vectorizer.fit_transform(preprocessed_docs_tmp)
        temp_vocabulary = set(vectorizer.get_feature_names_out())

        preprocessed_docs_tmp = [' '.join([w for w in doc.split() if w in temp_vocabulary])
                                 for doc in preprocessed_docs_tmp]

        preprocessed_docs, unpreprocessed_docs, retained_indices = [], [], []
        for i, doc in enumerate(preprocessed_docs_tmp):
            if len(doc) > 0 and len(doc) >= self.min_words:
                preprocessed_docs.append(doc)
                unpreprocessed_docs.append(self.documents[i])
                retained_indices.append(i)

        vocabulary = list(set([item for doc in preprocessed_docs for item in doc.split()]))

        return preprocessed_docs, unpreprocessed_docs, vocabulary, retained_indices

def get_bag_of_words(data, min_length):
    """
    Creates the bag of words
    """
    vect = [np.bincount(x[x != np.array(None)].astype('int'), minlength=min_length)
            for x in data if np.sum(x[x != np.array(None)]) != 0]

    vect = scipy.sparse.csr_matrix(vect)
    return vect


def bert_embeddings_from_file(text_file, sbert_model_to_load, batch_size=200, max_seq_length=None):
    """
    Creates SBERT Embeddings from an input file, assumes one document per line
    """

    model = SentenceTransformer(sbert_model_to_load)

    if max_seq_length is not None:
        model.max_seq_length = max_seq_length

    with open(text_file, encoding="utf-8") as filino:
        texts = list(map(lambda x: x, filino.readlines()))

    check_max_local_length(max_seq_length, texts)

    return np.array(model.encode(texts, show_progress_bar=True, batch_size=batch_size))


def bert_embeddings_from_list(texts, sbert_model_to_load, batch_size=200, max_seq_length=None):
    """
    Creates SBERT Embeddings from a list
    """
    model = SentenceTransformer(sbert_model_to_load)

    if max_seq_length is not None:
        model.max_seq_length = max_seq_length

    check_max_local_length(max_seq_length, texts)

    return np.array(model.encode(texts, show_progress_bar=True, batch_size=batch_size))


def check_max_local_length(max_seq_length, texts):
    max_local_length = np.max([len(t.split()) for t in texts])
    if max_local_length > max_seq_length:
        warnings.simplefilter('always', DeprecationWarning)


class TopicModelDataPreparation:

    def __init__(self, contextualized_model=None, show_warning=True, max_seq_length=128):
        self.contextualized_model = contextualized_model
        self.vocab = []
        self.id2token = {}
        self.vectorizer = None
        self.label_encoder = None
        self.show_warning = show_warning
        self.max_seq_length = max_seq_length

    def load(self, contextualized_embeddings, bow_embeddings, id2token, labels=None):
        return CTMDataset(
            X_contextual=contextualized_embeddings, X_bow=bow_embeddings, idx2token=id2token, labels=labels)

    def fit(self, text_for_contextual, text_for_bow, labels=None, custom_embeddings=None):
        """
        This method fits the vectorizer and gets the embeddings from the contextual model

        :param text_for_contextual: list of unpreprocessed documents to generate the contextualized embeddings
        :param text_for_bow: list of preprocessed documents for creating the bag-of-words
        :param custom_embeddings: np.ndarray type object to use custom embeddings (optional).
        :param labels: list of labels associated with each document (optional).
        """

        if custom_embeddings is not None:
            assert len(text_for_contextual) == len(custom_embeddings)

            if text_for_bow is not None:
                assert len(custom_embeddings) == len(text_for_bow)

            if type(custom_embeddings).__module__ != 'numpy':
                raise TypeError("contextualized_embeddings must be a numpy.ndarray type object")

        if text_for_bow is not None:
            assert len(text_for_contextual) == len(text_for_bow)

        if self.contextualized_model is None and custom_embeddings is None:
            raise Exception("A contextualized model or contextualized embeddings must be defined")

        # TODO: this count vectorizer removes tokens that have len = 1, might be unexpected for the users
        self.vectorizer = CountVectorizer()

        train_bow_embeddings = self.vectorizer.fit_transform(text_for_bow)

        # if the user is passing custom embeddings we don't need to create the embeddings using the model

        if custom_embeddings is None:
            train_contextualized_embeddings = bert_embeddings_from_list(
                text_for_contextual, sbert_model_to_load=self.contextualized_model, max_seq_length=self.max_seq_length)
        else:
            train_contextualized_embeddings = custom_embeddings
        self.vocab = self.vectorizer.get_feature_names_out()
        self.id2token = {k: v for k, v in zip(range(0, len(self.vocab)), self.vocab)}

        if labels:
            self.label_encoder = OneHotEncoder()
            encoded_labels = self.label_encoder.fit_transform(np.array([labels]).reshape(-1, 1))
        else:
            encoded_labels = None
        return CTMDataset(
            X_contextual=train_contextualized_embeddings, X_bow=train_bow_embeddings,
            idx2token=self.id2token, labels=encoded_labels)

    def transform(self, text_for_contextual, text_for_bow=None, custom_embeddings=None, labels=None):
        """
        This method create the input for the prediction. Essentially, it creates the embeddings with the contextualized
        model of choice and with trained vectorizer.

        If text_for_bow is missing, it should be because we are using ZeroShotTM

        :param text_for_contextual: list of unpreprocessed documents to generate the contextualized embeddings
        :param text_for_bow: list of preprocessed documents for creating the bag-of-words
        :param custom_embeddings: np.ndarray type object to use custom embeddings (optional).
        :param labels: list of labels associated with each document (optional).
        """

        if custom_embeddings is not None:
            assert len(text_for_contextual) == len(custom_embeddings)

            if text_for_bow is not None:
                assert len(custom_embeddings) == len(text_for_bow)

        if text_for_bow is not None:
            assert len(text_for_contextual) == len(text_for_bow)

        if self.contextualized_model is None:
            raise Exception("You should define a contextualized model if you want to create the embeddings")

        if text_for_bow is not None:
            test_bow_embeddings = self.vectorizer.transform(text_for_bow)
        else:
            # dummy matrix
            if self.show_warning:
                warnings.simplefilter('always', DeprecationWarning)
                warnings.warn(
                    "The method did not have in input the text_for_bow parameter. This IS EXPECTED if you "
                    "are using ZeroShotTM in a cross-lingual setting")

            # we just need an object that is matrix-like so that pytorch does not complain
            test_bow_embeddings = scipy.sparse.csr_matrix(np.zeros((len(text_for_contextual), 1)))

        if custom_embeddings is None:
            test_contextualized_embeddings = bert_embeddings_from_list(
                text_for_contextual, sbert_model_to_load=self.contextualized_model, max_seq_length=self.max_seq_length)
        else:
            test_contextualized_embeddings = custom_embeddings

        if labels:
            encoded_labels = self.label_encoder.transform(np.array([labels]).reshape(-1, 1))
        else:
            encoded_labels = None

        return CTMDataset(X_contextual=test_contextualized_embeddings, X_bow=test_bow_embeddings,
                          idx2token=self.id2token, labels=encoded_labels)

documents = [line.strip() for line in (message + test_message) if not isinstance(line, float)]
test_documents = [line.strip() for line in test_message if not isinstance(line, float)]

sp_train = WhiteSpacePreprocessingStopwords(documents, stopwords_list=stopwords)
preprocessed_documents, unpreprocessed_corpus, vocab, retained_indices = sp_train.preprocess()
labels = pd.concat([train, test]).reset_index()['label'][retained_indices]

sp_test = WhiteSpacePreprocessingStopwords(test_documents, stopwords_list=stopwords)
test_preprocessed_documents, test_unpreprocessed_corpus, test_vocab, test_retained_indices = sp_test.preprocess()
test_labels = test['label'][test_retained_indices]

mallet_stopwords = []
loc3 = "mallet_stopwords.txt"
with open(loc3) as f:
  for line in f:
    mallet_stopwords.append(line.strip())

mallet_stopwords = mallet_stopwords[0].split(",")
mallet_stopwords = [word.strip() for word in mallet_stopwords if word.strip()] + [","]

def preprocess(documents, topics, weights):
    """
    PARAMS
        documents: list of documents (each element in the list is a string) that the topics were extracted from
        topics: list of list of topics from the model of choice
        weights: list of dictionary mappings (word: weight)

    RETURN
        topic distribution
    """

    # Initialize distribution matrix
    distribution = np.zeros((len(documents), len(topics)))

    for i, document in enumerate(documents):
        # Preprocess document
        document = document.translate(str.maketrans('', '', string.punctuation)) # removing periods, commas, etc
        document = document.split(' ') # split on spaces
        document = [word.lower() for word in document if len(word.lower()) > 0] # lower case everything since all topics are lower case

        for j, loglik_dict in enumerate(weights):
            distribution[i][j] = np.sum([0 if word not in loglik_dict else loglik_dict[word] for word in document]) # if word exists then its weight else 0

        # Normalize
        distribution[i] /= (len(document) + 2) # +2 for some weird reason ?
        
    return distribution

hyperparams = {
  'num_topics': num_topics_list,
  'num_top_words': num_topics_word,
  'percentage_of_dataset': 1,
  'text_column': text_column,
  'label_column': label_column,
  'id_column': id_column,
}

for run in range(1, 6):
    
    print("Start run " + str(run))

    # make new directory
    try:
        os.makedirs(folder + 'NMF')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # make new run directory
    try:
        os.makedirs(folder + 'NMF/run_' + str(run))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # write hyperparams to file
    import json
    with open(folder + 'NMF/run_' + str(run) + '/hyperparams.txt', 'w') as f:
        f.write(json.dumps(hyperparams))

    # write stopwords to file
    with open(folder + 'NMF/run_' + str(run) + '/stopwords.txt', 'w') as f:
        for item in mallet_stopwords:
            f.write(item + ", ")

    for num_topics in num_topics_list:
        print("Start " + str(num_topics))

        vectorizer = TfidfVectorizer(max_features=num_topics_word, min_df=10, stop_words='english')
        X = vectorizer.fit_transform(documents + test_documents)
        words = np.array(vectorizer.get_feature_names_out())
        
        nmf = NMF(n_components=num_topics, solver="mu")
        W = nmf.fit_transform(X)
        H = nmf.components_

        nmf_loglik = []
        nmf_topics = []
        for i in range(len(H)):
            temp_words = words[H[i].argsort()]
            temp_weights = sorted(H[i])

            temp_dict = {}
            temp_topics = []

            for i in range(len(temp_words)):
                if temp_words[i] not in mallet_stopwords:
                    temp_dict[temp_words[i]] = temp_weights[i]
                    temp_topics.append(temp_words[i])
        
            nmf_loglik.append(temp_dict)
            nmf_topics.append(temp_topics)
       
        X = preprocess([line.strip() for line in message if not isinstance(line, float)], nmf_topics, nmf_loglik)
        X_test = preprocess([line.strip() for line in test_message if not isinstance(line, float)], nmf_topics, nmf_loglik)

        with open(folder + 'NMF/run_' + str(run) + '/topics_' + str(num_topics) + '.txt', 'w') as f:
            for sublist in nmf_topics:
                for i in sublist:
                    if i not in mallet_stopwords:
                        f.write(i + ", ")
                f.write("\n")
                
        with open(folder + 'NMF/run_' + str(run) + '/labels.pkl', 'wb') as f:
            pickle.dump(labels, f)
        
        with open(folder + 'NMF/run_' + str(run) + '/train.pkl', 'wb') as f:
            pickle.dump(train, f)
        
        with open(folder + 'NMF/run_' + str(run) + '/test.pkl', 'wb') as f:
            pickle.dump(test, f)
        
        with open(folder + 'NMF/run_' + str(run) + '/nmf_loglik.pkl', 'wb') as f:
            pickle.dump(nmf_loglik, f)
        
        # Creating avg_topic_distributions
        if folder[:-1] in ['fb', 'qualtrics', 'twitter', 'facebook_2020', 'facebook']:
            model = 'NMF'
            temp = pd.concat([train, test]).reset_index(drop=True) # concatenating train and test datasets
            df = pd.read_csv(dataset_path) # loading in actual dataset

            test_distribution = X_test
            train_distribution = X
            distribution = np.concatenate([train_distribution, test_distribution]) # concatenating train and test distributions

            merged = pd.merge(temp, df, how='inner', left_on = 'message_id', right_on = 'Unnamed: 0')[['message_id_x', 'message_x', 'user_id']]

            distribution_df = pd.concat([pd.DataFrame(distribution), merged[['user_id']]], axis = 1)
            user_distributions = distribution_df.groupby('user_id').mean().reset_index(drop=True)
            user_distributions_arr = [row[1].values for row in user_distributions.iterrows()]

            with open(folder + model + '/run_' + str(run) + "/" + model + '_avg_topic_distribution.pkl', 'wb') as f:
                pickle.dump(user_distributions_arr, f)
        
        else:
            with open(folder + 'NMF/run_' + str(run) + '/NMF_topic_distribution_test.pkl', 'wb') as f:
                pickle.dump(X_test, f)
            
            with open(folder + 'NMF/run_' + str(run) + '/NMF_topic_distribution_train.pkl', 'wb') as f:
                pickle.dump(X, f)

        print("End " + str(num_topics))
        print()