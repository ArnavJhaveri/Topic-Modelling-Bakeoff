import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import abc
import re
import itertools
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
import csv
import pickle

def proportion_unique_words(topics, topk=10):
    if topk > len(topics[0]):
        raise Exception('Words in topics are less than '+str(topk))
    else:
        unique_words = set()
        for topic in topics:
            unique_words = unique_words.union(set(topic[:topk]))
        puw = len(unique_words) / (topk * len(topics))
        return puw

class Coherence(abc.ABC):
    def __init__(self, topics, texts):
        self.topics = topics
        self.texts = texts
        self.dictionary = Dictionary(self.texts)

    @abc.abstractmethod
    def score(self):
        pass

class CoherenceNPMI(Coherence):
    def __init__(self, topics, texts):
        super().__init__(topics, texts)

    def score(self, topk=10, per_topic=False):
        if topk > len(self.topics[0]):
            raise Exception('Words in topics are less than topk')
        else:
            npmi = CoherenceModel(
                topics=self.topics, texts=self.texts,
                dictionary=self.dictionary,
                coherence='c_npmi', topn=topk)
            if per_topic:
                return npmi.get_coherence_per_topic()
            else:
                return npmi.get_coherence()

dataset = 'facebook_2020'

print(dataset)

df = pd.read_csv('data/' + dataset + '.csv')
documents = [line.strip() for line in (df['message']) if not isinstance(line, float)]

model_scores = {} # tuple of model : (avg diversity, avg coherence)

pattern = re.compile('\w+')
texts = [pattern.findall(t.lower()) for t in documents]

for model in ['Mallet_LDA', 'CTM', 'BERTopic', 'NMF']:
    print(model)

    diversities = []
    coherences = []

    for run in range(1, 6):
        print(run)

        topics = []
        with open(dataset + '/' + model + '/run_' + str(run) + '/' + 'topics_100.txt', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                topic_list = [item.strip() for item in row if item.strip()]
                topics.append(topic_list)
        
        warnings.filterwarnings('ignore')
        topic_diversity = proportion_unique_words(topics)
        print(topic_diversity)
        coherence = CoherenceNPMI(texts=texts, topics=topics).score()
        print(coherence)

        diversities.append(topic_diversity)
        coherences.append(coherence)
    
    model_scores[model] = (np.mean(diversities), np.mean(coherences))
    print(np.mean(diversities))
    print(np.mean(coherences))

print(model_scores)
with open(dataset + '_model_scores.pkl', 'wb') as f:
    pickle.dump(model_scores, f)