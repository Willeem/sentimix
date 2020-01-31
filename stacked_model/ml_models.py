#!/usr/bin/env python3

from sklearn import svm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC, NuSVC
from textblob import TextBlob

class FeaturesExtractor(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, articles):
        # Possible transformations: 
        #   - POS-tags
        #   - Named entities
        #   - BERT (not yet fully integrated)
        #   - Context independent embeddings (normally used by statistical ML models)
        #   - Character n-grams 
        features = {}
        features['text'] = [words for (words, x_high, sentence, x_tr, x_tr_high, x_tr_sentence) in articles]
        features['text_high'] = [x_high for (words, x_high, sentence, x_tr, x_tr_high, x_tr_sentence) in articles]
        features['text_translated'] = [x_tr for (words, x_high, sentence, x_tr, x_tr_high, x_tr_sentence) in articles]
        features['text_translated_high'] = [x_tr_high for (words, x_high, sentence, x_tr, x_tr_high, x_tr_sentence) in articles]
        features['text_ngram'] =[sentence for (words, x_high, sentence, x_tr, x_tr_high, x_tr_sentence) in articles]
        features['text_ngram_translated'] = [x_tr_sentence for (words, x_high, sentence, x_tr, x_tr_high, x_tr_sentence) in articles]

        return features


class SentimentContinuous(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, tweets):
        sentiment = []
        for tweet in tweets:
            blob = TextBlob(tweet)
            sentiment.append([blob.sentiment.polarity])
        return sentiment


class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


def identity(x):
    return x


def model_words():
    '''
    The model + pipeline for features extracted from the text
    '''
    clfs = [LinearSVC(), svm.SVC(kernel='linear', C=1.0), PassiveAggressiveClassifier(max_iter=250, tol=1e-3),
    PassiveAggressiveClassifier(C=0.001, class_weight="balanced", fit_intercept=False, loss="squared_hinge", max_iter=7500)]
    
    classifier = Pipeline([
        # Extract the features
        ('features', FeaturesExtractor()),
        # Use FeatureUnion to combine the features from subject and body
        ('union', FeatureUnion(
            transformer_list = [
                # Pipeline bag-of-words model 
                ('words', Pipeline([
                    ('selector', ItemSelector(key='text')),
                    ('tfidf', TfidfVectorizer(preprocessor = identity, tokenizer = identity, 
                                              max_df = .2)),
                    ('chi-square', SelectKBest(chi2, 300)), #3000)),
                ])),

                # Pipeline for high info words bag-of-words model 
                ('text_high', Pipeline([
                    ('selector', ItemSelector(key='text_high')),
                    ('tfidf', TfidfVectorizer(preprocessor = identity, tokenizer = identity, 
                                              max_df = .2)),
                ])),

                # ('char_n_grams', Pipeline([
                #     ('selector', ItemSelector(key='text_ngram')),
                #     ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(2,6)))
                # ])),
                
                ('word_n_grams', Pipeline([
                    ('selector', ItemSelector(key='text_ngram')),
                    ('tfidf', TfidfVectorizer(analyzer='word', ngram_range=(1,3)))
                ])),

                # ('sentiment_cont', Pipeline([
                #     ('selector', ItemSelector(key='text_ngram')),
                #     ('feature', SentimentContinuous())
                # ])),

            ],
        )),
        # Use a classifier on the combined features
        ('clf', clfs[3]),
    ])
    return classifier


def model_translated_english():
    '''
    The model + pipeline for features extracted from the translated text (Spanish to English)
    '''
    clfs = [LinearSVC(), svm.SVC(kernel='linear', C=1.0), PassiveAggressiveClassifier(max_iter=250, tol=1e-3)]
    
    classifier = Pipeline([
        # Extract the features
        ('features', FeaturesExtractor()),
        # Use FeatureUnion to combine the features from subject and body
        ('union', FeatureUnion(
            transformer_list = [
                # Pipeline bag-of-words model 
                ('words', Pipeline([
                    ('selector', ItemSelector(key='text_translated')),
                    ('tfidf', TfidfVectorizer(preprocessor = identity, tokenizer = identity, 
                                              max_df = .2)),
                    ('chi-square', SelectKBest(chi2, 300)), #3000)),
                ])),

                # Pipeline for high info words bag-of-words model 
                ('text_high', Pipeline([
                    ('selector', ItemSelector(key='text_translated_high')),
                    ('tfidf', TfidfVectorizer(preprocessor = identity, tokenizer = identity, 
                                              max_df = .2)),
                ])),
                 ('char_n_grams', Pipeline([
                    ('selector', ItemSelector(key='text_ngram_translated')),
                    ('tfidf', TfidfVectorizer(analyzer='char',ngram_range=(2,6)))
                ])),
                ('word_n_grams', Pipeline([
                    ('selector', ItemSelector(key='text_ngram_translated')),
                    ('tfidf', TfidfVectorizer(analyzer='word',ngram_range=(1,3)))
                ])),
            ],
        )),
        # Use a classifier on the combined features
        ('clf', clfs[2]),
    ])
    return classifier


def model_meta():
    '''
    The final meta classifier using the outputs from the other models as input
    '''
    # return PassiveAggressiveClassifier(max_iter=250, tol=1e-3)
    return svm.SVC(kernel='linear', C=1.0)
