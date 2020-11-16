"""
Main script to run ETM
"""
import os
import argparse
import pandas as pd

from src.wrapper import end2end
from src.eval.metrics import get_topic_diversity, get_topic_coherence


# parse user args
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--corpus", default='data/movies/movies.csv', help="directory of corpus")
parser.add_argument("-s", "--savedmodel", default = 'savedmodel', help="directory of saved model")
parser.add_argument("-n", "--ntopics", default = 7, help="number of topics")
args = parser.parse_args()


# read data
movies = pd.read_csv(args.corpus)
train_data = movies.overview.tolist()


# train or load data
if args.savedmodel not in os.listdir():
    etm = end2end(num_topics=args.ntopics, min_df=0.001)
    etm.fit_topic_model(movies.overview.tolist(), 
                        magnitude = 'magnitude/GoogleNews-vectors-negative300.magnitude',
                        freeze_topic_prior=True,
                        epochs=500,
                        verbose=1)
    etm.save_model(args.savedmodel)
else:
    etm = end2end(num_topics=args.ntopics, min_df=0.001)
    etm.load_model(args.savedmodel)


# interpret
print( "\nInterpreting the topics" )
etm.top_k_words(n=10)
# Topic 0: ['love', 'young', 'woman', 'family', 'wife', 'father', 'man', 'girl', 'daughter']
# Topic 1: ['world', 'film', 'war', 'story', 'american', 'set', 'movie', 'based', 'series']
# Topic 2: ['find', 'finds', 'life', 'takes', 'meets', 'begins', 'discovers', 'home', 'decides']
# Topic 3: ['york', 'john', 'dr', 'jack', 'tom', 'de', 'los', 'king', 'peter']
# Topic 4: ['man', 'police', 'murder', 'gang', 'killer', 'crime', 'named', 'agent', 'killed']
# Topic 5: ['group', 'team', 'earth', 'save', 'mission', 'battle', 'evil', 'army', 'fight']
# Topic 6: ['time', 'years', 'town', 'life', 'friends', 'school', 'day', 'back', 'city']
