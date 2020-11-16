from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pickle
import random
import numbers
from scipy import sparse
import itertools
from scipy.io import savemat, loadmat
import re
import string

from .stops import stops

class Preprocessor(CountVectorizer):
    
    def __init__(self, min_df=0.005, max_df=0.7):
        super().__init__(min_df=min_df, max_df=max_df, stop_words=stops)
        
    def clean_doc(self, doc):
        """
        Removes words with puncutations / numbers and one letter words
        """
        list_of_tokens = re.findall(r'''[\w']+|[.,!?;-~{}`´_<=>:/@*()&'$%#"]''', doc)

        def contains_punctuation(w):
            return any(char in string.punctuation for char in w)
        def contains_numeric(w):
            return any(char.isdigit() for char in w)

        # filter out words with punctuation, eg "where's"
        list_of_tokens = [token.lower() for token in list_of_tokens if not contains_punctuation(token)]
        # filter out words with numbers, eg "rac3"
        list_of_tokens = [token for token in list_of_tokens if not contains_numeric(token)]
        # remove one letter words
        list_of_tokens = [token for token in list_of_tokens if len(token)>1]

        return ' '.join(list_of_tokens)
    
    def clean_corpus(self, corpus):
        """
        Cleans iterable of string docs
        """
        return [self.clean_doc(doc) for doc in corpus]
    
    def fit_transform(self, raw_documents, y=None):
        """Learn the vocabulary dictionary and return term-document matrix.
        This is equivalent to fit followed by transform, but more efficiently
        implemented.
        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.
        Returns
        -------
        X : array, [n_samples, n_features]
            Document-term matrix.
        """
        # We intentionally don't call the transform method to make
        # fit_transform overridable without unwanted side effects in
        # TfidfVectorizer.
        if isinstance(raw_documents, str):
            raise ValueError(
                "Iterable over raw text documents expected, "
                "string object received.")

        self._validate_params()
        self._validate_vocabulary()
        max_df = self.max_df
        min_df = self.min_df
        max_features = self.max_features

        vocabulary, X = self._count_vocab(self.clean_corpus(raw_documents),
                                          self.fixed_vocabulary_)

        if self.binary:
            X.data.fill(1)

        if not self.fixed_vocabulary_:
            X = self._sort_features(X, vocabulary)

            n_doc = X.shape[0]
            max_doc_count = (max_df
                             if isinstance(max_df, numbers.Integral)
                             else max_df * n_doc)
            min_doc_count = (min_df
                             if isinstance(min_df, numbers.Integral)
                             else min_df * n_doc)
            if max_doc_count < min_doc_count:
                raise ValueError(
                    "max_df corresponds to < documents than min_df")
            X, self.stop_words_ = self._limit_features(X, vocabulary,
                                                       max_doc_count,
                                                       min_doc_count,
                                                       max_features)

            self.vocabulary_ = vocabulary

        return X
    
    def _transform(self, raw_documents):
        """Transform documents to document-term matrix.
        Extract token counts out of raw text documents using the vocabulary
        fitted with fit or the one provided to the constructor.
        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.
        Returns
        -------
        X : sparse matrix, [n_samples, n_features]
            Document-term matrix.
        """
        if isinstance(raw_documents, str):
            raise ValueError(
                "Iterable over raw text documents expected, "
                "string object received.")
        self._check_vocabulary()

        # use the same matrix-building strategy as fit_transform
        _, X = self._count_vocab(raw_documents, fixed_vocab=True)
        if self.binary:
            X.data.fill(1)
        return X
    
    def transform(self, raw_documents, y=None):
        return self._transform(self.clean_corpus(raw_documents))
    