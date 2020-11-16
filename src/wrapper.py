"""
Wrapper class will contain training, inference and interpretation
"""
import os 
import sys
import pickle 
import math 
import random 

import pandas as pd
import numpy as np 
from joblib import dump, load
import scipy.io
import torch
from torch import nn, optim
from torch.nn import functional as F
import pymagnitude
from pymagnitude import Magnitude

from .models.dan_etm import DAN_ETM
from .preprocessor import Preprocessor
from .eval.metrics import get_topic_coherence, get_topic_diversity
from .interpret.topic_model import nearest_neighbors, visualize_topic_embeddings
from .train.topic_model import fit_topic_model
from .train.classifier import train_classifier

np.random.seed(2019)
torch.manual_seed(2019)
if torch.cuda.is_available():
    torch.cuda.manual_seed(2019)
    
class end2end:
    def __init__(self, num_topics = 50, 
                 min_df=0.001, max_df=0.7,
                 lr = 0.005, wdecay=1.2e-6,
                 device='cuda'
                 ):

        self.num_topics = num_topics
        self.prep = Preprocessor(min_df=min_df, max_df=max_df)
        
        self.lr = lr
        self.wdecay = wdecay

        self.device = torch.device("cuda" if torch.cuda.is_available() & (device == 'cuda') else "cpu")
        self.is_fitted=False


    """
    FITTING FUNCTIONS
    """
    def sparse2torch(self, sparse):
        return torch.from_numpy( sparse.toarray() ).float().to(self.device)

    def fit_preprocess(self, train_data, verbose=1):
        """
        Fit preprocessor
        
        args:
        -----
            train_data: (iterable of str)
        """
        bow_train = self.prep.fit_transform(train_data)
        bow_train = self.sparse2torch(bow_train)
        self.bow_train = bow_train[~(bow_train.sum(1)==0)] # remove 0 rows, eg preprocessor cant find word
        
        self.vocab = self.prep.get_feature_names()
        if verbose: print(f"Preprocessor: {len(self.vocab)} words saved")
        
        return self.bow_train

    def preprocess(self, iterable_of_text):
        """
        Preprocess an iterable of text into a torch array
        """
        bow = self.prep.transform(iterable_of_text)
        bow = self.sparse2torch(bow)
        bow[(bow.sum(1)==0)] = 1/bow.size()[1]
        return bow

    def fit_topic_model(self, train_data, 
                        magnitude=None, doc_encoder_dim=800, doc_encoder_act=nn.ReLU(), enc_drop=0, clip=0,
                        seed_topic_list=None, freeze_topic_prior=False,
                        epochs = 1000, batchsize=1000, 
                        log_interval=2, verbose=1):
        """
        Train model and preprocessor

        args:
        ----
            train_data: (iterable of str) training data
            epochs: (int) rounds to train over the training data
            batchsize: (int) batch size of the training set
            verbose: (int / bool) to display training logs

        Example seed:
        ------------
        # https://github.com/vi3k6i5/GuidedLDA
        seed_topic_list = [['game', 'team', 'win', 'player', 'season', 'second', 'victory'],
                            ['percent', 'company', 'market', 'price', 'sell', 'business', 'stock', 'share'],
                            ['music', 'write', 'art', 'book', 'world', 'film'],
                            ['political', 'government', 'leader', 'official', 'state', 'country', 'american',
                            'case', 'law', 'police', 'charge', 'officer', 'kill', 'arrest', 'lawyer']]
        """
        fit_topic_model(self, train_data, 
                        magnitude, doc_encoder_dim, doc_encoder_act, enc_drop, clip,
                        seed_topic_list, freeze_topic_prior,
                        epochs, batchsize, 
                        log_interval, verbose)

        return self

    def get_magnitude(self, magnitude):
        if type(magnitude)==str:
            magnitude = Magnitude(magnitude)
        elif type(magnitude)==pymagnitude.Magnitude:
            pass
        else:
            raise ValueError("magnitude type not recognized")
        return magnitude

    def get_pretrained_word_embed(self, vocab, magnitude='GoogleNews-vectors-negative300.magnitude', verbose=1):
        """
        loads word embeddings from magnitude into pytorch

        Return:
        -----
            pretrained_word_embed: torch.Tensor of size (Vocab x word embedding size)
            unknown_words: list of unknown words that are not in the prefit embeddings
        """
        magnitude = Magnitude(magnitude)
        
        pretrained_word_embed = [torch.from_numpy(magnitude.query(word)).type(torch.cuda.FloatTensor).view(1,-1)
                                for word in vocab]
        pretrained_word_embed = torch.cat(pretrained_word_embed, axis=0)
        
        self.unknown_words = [word for word in vocab if word not in magnitude]
        n_unknown_words = len(self.unknown_words)
        if verbose: print(f"Unknown words: {n_unknown_words} / {len(vocab)}")
        
        return pretrained_word_embed


    """
    Inference methods
    """
    def predict_topic(self, iterable_of_documents):
        """
        Classify the documents, by returning the probabilities
        """
        data_batch = self.preprocess(iterable_of_documents)
        yhat = F.sigmoid( self.model.encode(data_batch)[0] )
        yhat = yhat.round()
        yhat[yhat==-1] = 0

        return yhat


    """
    Downstream classifier
    """
    def train_classifier(self, train_data, labels, layer_num=2, freeze_enc=True, epochs=1000, batchsize=1000, verbose=1):
        """
        
        args:
        -----
            train_data: (list) contains corpus
            labels: (np.array) 
        """
        # initialize new weights
        train_classifier(self, train_data, labels, layer_num, freeze_enc, epochs, batchsize, verbose)

        return self


    def predict_proba(self, list_of_strings):
        bow = self.preprocess(list_of_strings)
        yhat = self.model.classify(bow)
        return yhat

    def predict(self, list_of_strings):
        yhat = self.predict_proba(list_of_strings)
        return torch.round(yhat)


    """
    Model persistence
    """
    def save_model(self, ckpt = 'model_ckpt'):

        # ensure folder exists
        if not os.path.isdir(ckpt):
            os.mkdir(ckpt)

        # save model
        with open(f"{ckpt}/model", 'wb') as f:
            torch.save(self.model, f)
        
        # save preprocessor
        dump(self.prep, f"{ckpt}/prep.joblib") 

        # save training data
        torch.save(self.bow_train,  f"{ckpt}/bow_train.pt")

        print(f"success: model and preprocessor saved in {ckpt}")

    def load_model(self, ckpt='model_ckpt'):

        # actual pytorch model
        with open(f"{ckpt}/model", 'rb') as f:
            self.model = torch.load(f)
        self.model = self.model.to(self.device)

        # preprocessor and vocab
        self.prep = load(f"{ckpt}/prep.joblib") 
        self.vocab = self.prep.get_feature_names()

        # load training data
        self.bow_train = torch.load(f"{ckpt}/bow_train.pt")

        self.is_fitted = True
        print(f"success: model and preprocessor loaded from {ckpt}")


    """
    Interpretive methods
    """
    def get_word_representation(self, word_string):
        """
        Get word emebdding 
        """
        vocab_idx = self.vocab.index(word_string)
        word_representation = self.model.word_embed[vocab_idx]

        topic_similarity = torch.nn.CosineSimilarity()(self.model.topic_embed.weight, 
                                                       word_representation.view(1,-1))#.sort()
        return word_representation, topic_similarity

    def visualize_nearest_neighbours(self, queries):
        """
        sample queries:
        ---------------
            queries = ['andrew', 'computer', 'sports', 'religion', 'man', 'love', 
                        'intelligence', 'money', 'politics', 'health', 'people', 'family']
        
        sample output:
        --------------
            vectors:  (3072, 300)
            query:  (300,)
            word: sports .. etm neighbors: ['sports', 'sport', 'rangers', 'clutch', 'ticket',  
                                            'player', 'hockey', 'nhl', 'stats', 'coach', 'calgary',
                                            'dog', 'hits', 'hitting', 'philadelphia', 'pitch', 'club', 
                                            'espn', 'playoff', 'leafs']
            vectors:  (3072, 300)
            query:  (300,)
            word: religion .. etm neighbors: ['religion', 'religious', 'islamic', 'christians', 
                                              'islam', 'jews', 'church', 'muslim', 'religions', 
                                              'moral', 'christ', 'beings', 'sandvik', 'innocent', 
                                              'atheists', 'koresh', 'hatred', 'jesus', 'muslims', 
                                              'satan']
        """
        for word in queries:
            print('word: {} .. neighbors: {}'.format(
                word, nearest_neighbors(word, self.model.word_embed, self.prep.get_feature_names())))
        print('#'*100)


    def top_k_words(self, n=10):
        """
        Visualize the top k words for each topic
        """
        with torch.no_grad():
            gammas = self.model.get_word_topic_matrix()
            for k in range(self.num_topics):
                gamma = gammas[k]
                top_words = list(gamma.cpu().numpy().argsort()[-n+1:][::-1])
                topic_words = [self.vocab[a] for a in top_words]
                print('Topic {}: {}'.format(k, topic_words))

    def evaluate(self):
        """
        Return topic diversity, topic coherence and the product of the two
        """
        word_topic_matrix = self.model.get_word_topic_matrix()

        td = get_topic_diversity(word_topic_matrix, 25)
        tc = get_topic_coherence(self.bow_train, word_topic_matrix)

        return td, tc, td*tc

    def visualize_topic_embeddings(self, topic_idx=0, n=15, alpha=0.7, size=35):
        """
        Visualize topics and closest words in 2D scatterplot (TSNE)

        args:
        -----
            topic_idx: (int) index of topic
            n: (int) top words of that topic
            alpha: (float) alpha of the words in scatter plots 
                           Not the points but the word annotations
            size: (int) font size of annotations
        """
        visualize_topic_embeddings(self, topic_idx, n, alpha, size)