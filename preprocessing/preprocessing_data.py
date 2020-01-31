#!/usr/bin/env python3

import collections
import json
import pickle
import re
import sys

import pandas as pd
import progressbar
import spacy
from nltk.corpus import stopwords
from nltk.metrics import BigramAssocMeasures
from nltk.probability import ConditionalFreqDist, FreqDist
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction import stop_words

nlp = spacy.load("en_core_web_sm")
snow = SnowballStemmer('english')

def uniques(words, badwords=False):
    '''
    Returns set of unique words, and filters badwords from them
    '''
    if not badwords:
        return set(words)
    return set(words) - set(badwords)


def test_eof_correct(corpus_file, conll, title):
    '''
    Ensure there are two newlines at the end-of-file (EOF)
    '''
    with open(corpus_file, encoding='utf-8') as f:
        if f.read()[-2:] != "\n\n":
            with open(corpus_file, "a") as file:
                file.write("\n")
            read_corpus(corpus_file, conll=conll, title=title)


def read_corpus(corpus_file, conll=True, pickle=False, title=""):
    '''
    Create a bag-of-words and labels from a file
    '''
    print(f"\n#### Reading in data [{title}]")
    nltk_stopword_set = set(stopwords.words('english')) #179 words
    scikit_stopword_set = set(stop_words.ENGLISH_STOP_WORDS) #318 words
    union_stopword_set = nltk_stopword_set | scikit_stopword_set # 378 words
    labels, sentence, documents = [], [], []

    if conll:
        test_eof_correct(corpus_file, conll, title)
        with open(corpus_file, encoding='utf-8') as f:
            for line in f:
                line = line.split()
                if len(line) == 3:
                    sentiment = line[2]
                    labels.append(sentiment)
                elif line != []:
                    sentence.append(line[0])
                else:
                    documents.append(sentence)
                    sentence = []
        return documents, labels
    elif pickle:
        # hier pickle
        return documents, labels
    else:
        with open(corpus_file) as f:
            for line in f:
                documents.append(line.strip().split())
        return documents, labels


def preprocessing(documents):
    '''
    Some simple pre-processing, for example changing all numbers to "number"
    '''
    for i in range(len(documents)):
        documents[i] = [re.sub(r'(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z0-9]+[A-Za-z0-9-_]+)', 'usrname', doc) for doc in documents[i]]
        documents[i] = [re.sub(r'(?<=^|(?<=[^a-zA-Z0-9-_\.]))#([A-Za-z0-9]+[A-Za-z0-9-_]+)', 'hashtag', doc) for doc in documents[i]]
        documents[i] = [re.sub(r'^([\s\d]+)$','number', doc) for doc in documents[i]]
        documents[i] = [re.sub(r'<[^<>]+>','', doc) for doc in documents[i]]
        documents[i] = [re.sub(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\-_=#])*','httpaddr', doc) for doc in documents[i]]
        documents[i] = [re.sub(r'[\w\.-]+@[\w\.-]+','emailaddr', doc) for doc in documents[i]]
        documents[i] = [re.sub(r'[$|¢|£|¤|¥|֏|؋|৲|৳|৻|૱|௹|฿|៛|₠|-|₽|꠸|﷼|﹩|＄|￠|￡|￥|￦]\d+([., ]?\d*)*', 'money', doc) for doc in documents[i]]
        # documents[i] = [re.sub(r'[^a-zA-Z0-9]', ' ', doc) for doc in documents[i]]
        documents[i] = [doc.strip() for doc in documents[i] if doc != " " and len(doc)>1]


def get_high_information_words(labelled_words, score_fn=BigramAssocMeasures.chi_sq, min_score=5):
    '''
    Gets the high information words using chi square measure
    '''
    word_fd = FreqDist()
    label_word_fd = ConditionalFreqDist()

    for label, words in labelled_words:
        for word in words:
            word_fd[word] += 1
            label_word_fd[label][word] += 1

    n_xx = label_word_fd.N()
    high_info_words = set()

    for label in label_word_fd.conditions():
        n_xi = label_word_fd[label].N()
        word_scores = collections.defaultdict(int)

        for word, n_ii in label_word_fd[label].items():
            n_ix = word_fd[word]
            score = score_fn(n_ii, (n_ix, n_xi), n_xx)
            word_scores[word] = score

        bestwords = [word for word, score in word_scores.items() if score >= min_score]
        high_info_words |= set(bestwords)

    return high_info_words


def high_information_words(X, y, title):
    '''
    Get and display info on high info words
    '''
    print(f"#### OBTAINING HIGH INFO WORDS [{title}]...")

    labelled_words = []
    amount_words = 0
    distinct_words = set()
    for words, genre in zip(X, y):
        labelled_words.append((genre, words))
        amount_words += len(words)
        for word in words:
            distinct_words.add(word)

    c = 4
    if title == "Test":
        c = 2.5
    elif title == "Train":
        c = 3
    high_info_words = set(get_high_information_words(labelled_words, BigramAssocMeasures.chi_sq, c)) # 7

    print("\tNumber of words in the data: %i" % amount_words)
    print("\tNumber of distinct words in the data: %i" % len(distinct_words))
    print("\tNumber of distinct 'high-information' words in the data: %i" % len(high_info_words))

    return high_info_words


def return_high_info(X, y, title="data"):
    '''
    Return list of high information words per document
    '''
    try:
        high_info_words = high_information_words(X, y, title)

        X_high_info = []
        for bag in X:
            new_bag = []
            for words in bag:
                if words in high_info_words:
                    new_bag.append(words)
            X_high_info.append(new_bag)
    except ZeroDivisionError:
        print("Not enough information too get high information words, please try again with more files.", file=sys.stderr)
        X_high_info = X
    return X_high_info


def return_named_ent(X, title="data"):
    '''
    Return list of named entities per document
    '''
    print(f"\n#### RETRIEVING NAMED ENTITIES TAGS [{title}]...")
    named_ent = []
    for bag, _ in zip(X, progressbar.progressbar(range(len(X)))):
        new_bag = []
        for ent in nlp(" ".join(bag)).ents:
            new_bag.append(ent.label_)
        named_ent.append(new_bag)
    return named_ent


def return_pos_tagged(X, title="data"):
    '''
    Return list of part-of-speech tags per document
    '''
    print(f"\n#### RETRIEVING PART-OF-SPEECH TAGS [{title}]...")
    pos_tag = []
    for bag, _ in zip(X, progressbar.progressbar(range(len(X)))):
        new_bag = []
        for pos in nlp(" ".join(bag)):
            new_bag.append(pos.tag_)
        pos_tag.append(new_bag)
    print()
    return named_ent


def get_sentences(X):
    return [" ".join(sentence) for sentence in X]


def read_normalised(file):
    with open(file, "r") as infile:
        normalised = json.load(infile)
    documents, labels = [], []
    for item in normalised:
        documents.append(" ".join([i[0] for i in item['text']]))
        labels.append(item['sentiment'])
    return documents, labels


def read_and_process(file, conll=True, pickle=False, Y_already=False, title=""):
    '''
    Reads in data from file to pandas dataframe, and preprocesses the data for the model
    '''
    X, Y = read_corpus(file, conll=conll, pickle=pickle, title=title)

    if Y_already:
        Y = Y_already

    preprocessing(X)

    X_high_info = return_high_info(X, Y, title)

    sentences = get_sentences(X)
    # X_pos = return_pos_tagged(X, "data")
    # X_ent = return_named_ent(X, "data")

    X = [(words, x_high, sentence) for words, x_high, sentence in zip(X, X_high_info, sentences)]

    return X, Y
