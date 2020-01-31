import json
import time
from collections import defaultdict, Counter

import requests
from lxml import html
import pandas as pd


def read_corpus_hindi(corpus_file):
    labels, sentence, documents = [], [], []
    with open(corpus_file, encoding='utf-8') as f:    
        lang_prev = ""
        part_dict = defaultdict(list)
        iteration = 0
        for line in f:
            line = line.split()
            if len(line) == 3:
                sentiment = line[2]
            elif line != []:
                try:
                    lang = line[1]
                except IndexError:
                    pass
                if lang != lang_prev and lang_prev != "":
                    iteration += 1
                    sentence.append(part_dict)
                    part_dict = defaultdict(list)
                keyname = lang + str(iteration)
                part_dict[keyname].append(line[0])
                try:
                    lang_prev = line[1]
                except IndexError:
                    pass
            else:
                sentence.append(part_dict)
                documents.append(sentence)
                labels.append(sentiment)
                sentence = []
                iteration = 0
                lang_prev = ""
                part_dict = defaultdict(list)
        return documents, labels


def read_corpus_spa(corpus_file):
    labels, sentence, documents = [], [], []
    with open(corpus_file, encoding='utf-8') as f:    
        lang_prev = ""
        part_dict = defaultdict(list)
        iteration = 0
        for line in f:
            line = line.split()
            if len(line) == 3:
                sentiment = line[2]
            elif line != []:
                lang = line[1]
                if lang != lang_prev and lang_prev != "":
                    iteration += 1
                    sentence.append(part_dict)
                    part_dict = defaultdict(list)
                keyname = lang + str(iteration)
                part_dict[keyname].append(line[0])
                lang_prev = line[1]
            else:
                sentence.append(part_dict)
                documents.append(sentence)
                labels.append(sentiment)
                sentence = []
                iteration = 0
                lang_prev = ""
                part_dict = defaultdict(list)
        return documents, labels

def divide_sentences_hindi(X,filetype):
    iteration = 0
    en, es, other = [], [], []
    for item in X:
        sentiment_dict = {}        
        for dictionary in item:
            iteration += 1
            lang = list(dictionary.keys())[0]
            if 'Eng' in lang:
                en.append(" ".join(list(dictionary.values())[0]))
    if filetype == 'train':
        with open('hing_to_normalise_train.txt', 'w') as outfile:
            for item in en:
                outfile.write(item+'\n')
    elif filetype == 'dev':
        with open('hing_to_normalise_dev.txt', 'w') as outfile:
            for item in en:
                outfile.write(item+'\n')


def divide_sentences_es(X, filetype):
    iteration = 0
    en, es, other = [], [], []
    for item in X:
        iteration += 1
        sentiment_dict = {}        
        for dictionary in item:
            if dictionary != {}:
                lang = list(dictionary.keys())[0]
                if 'lang1' in lang:
                    en.append(" ".join(list(dictionary.values())[0]))
                elif 'lang2' in lang:
                    es.append(" ".join(list(dictionary.values())[0]))
    if filetype == 'train':
        with open('to_normalise_train_en.txt', 'w') as outfile:
            for item in en:
                outfile.write(item+'\n')
        with open('to_normalise_train_es.txt', 'w') as outfile:
            for item in es:
                outfile.write(item+'\n')
    elif filetype == 'dev':
        with open('to_normalise_dev_en.txt', 'w') as outfile:
            for item in en:
                outfile.write(item+'\n')
        with open('to_normalise_dev_es.txt', 'w') as outfile:
            for item in es:
                outfile.write(item+'\n')
    


def concatenate_sentences_es(data,labels, state):
    with open(f'normalised_{state}_es.txt','r') as infile:
        es = infile.readlines()
    with open(f'normalised_{state}_en.txt','r') as infile:
        en = infile.readlines()
    i_en = 0
    i_es = 0
    missing_en = 0
    missing_es = 0
    sentiment_list = []
    for item, sentiment in zip(data,labels):  
        sentence = []
        for dictionary in item:
            lang = list(dictionary.keys())[0]
            if 'lang1' in lang:
                sentence.append((en[i_en].strip(),'lang1'))
                i_en += 1                
            elif 'lang2' in lang:
                sentence.append((es[i_es].strip(),'lang2'))
                i_es += 1
            else:
                sentence.append((" ".join(list(dictionary.values())[0]),'other'))
        flat_list = [item for item in sentence]
        sentiment_list.append({'sentiment':sentiment,'text':flat_list})
    return sentiment_list


def concatenate_sentences_hindi(data,labels,filename):
    with open(filename,'r') as infile:
        hindi = infile.readlines()
    i_hindi = 0
    sentiment_list = []
    for item, sentiment in zip(data,labels):
        sentence = [] 
        for dictionary in item:
            lang = list(dictionary.keys())[0]
            if 'Hin' in lang:
                sentence.append((" ".join(list(dictionary.values())[0]),'Hin'))
            elif 'Eng' in lang:
                sentence.append((hindi[i_hindi].strip(),'Eng'))
                i_hindi += 1
            else:
                sentence.append((" ".join(list(dictionary.values())[0]),'other'))
        flat_list = [item for item in sentence]
        sentiment_list.append({'sentiment':sentiment,'text':flat_list})
    return sentiment_list


def read_extra_data(inputfile):
    langs, words = [], []
    with open(inputfile, 'r', encoding='utf-8') as infile:
        for line in infile.readlines():
            line = line.split(',')
            if line[0] == 'comma':
                line[0] = ','
                line[2] = 'und'
            if line[-2] == '#VALUE!':
                line[-2] = 'und'
                line[0] = "'"
            elif line[0] == '#ERROR!':
                line[0] = '='
                line[-2] = 'und'
            langs.append(line[-2])
            words.append(line[0])
    lang_prev = ""
    documents, labels, sentence = [], [], []
    iteration = 0
    part_dict = defaultdict(list)
    for line, lang in zip(words,langs):
        if line != "":
            lang = lang if lang in ['en','und'] else 'es'
            if lang != lang_prev and lang_prev != "":
                iteration += 1
                sentence.append(part_dict)
                part_dict = defaultdict(list)
            keyname = lang + str(iteration)
            part_dict[keyname].append(line)
            lang_prev = lang
        else:
            print(part_dict)
            sentence.append(part_dict)
            documents.append(sentence)
            sentence = []
            iteration = 0
            lang_prev = ""
            part_dict = defaultdict(list)
    return documents, labels

def concatenate_sentences_extra(data,labels):
    with open('normalised_emoji_es.txt','r') as infile:
        es = infile.readlines()
    with open('normalised_emoji_en.txt','r') as infile:
        en = infile.readlines()
    i_en = 0
    i_es = 0
    missing_en = 0
    missing_es = 0
    sentiment_list = []
    for item in data:  
        sentence = []
        for dictionary in item:
            try:
                lang = list(dictionary.keys())[0]
            except:
                lang = ""
            if 'en' in lang:
                sentence.append((en[i_en].strip(),'en'))
                i_en += 1                
            elif 'es' in lang:
                sentence.append((es[i_es].strip(),'es'))
                i_es += 1
            else:
                try:
                    sentence.append((" ".join(list(dictionary.values())[0]),'other'))
                except:
                    pass
        flat_list = [item for item in sentence]
        sentiment_list.append(flat_list)
        print(len(sentiment_list))
    return sentiment_list

def concatenate_and_add_label(normalised_extra):
    for item in normalised_extra:
        print(item)

def main(inputfile,language,state):
    if language == 'hinglish':
        Xtrain, Ytrain = read_corpus_hindi(inputfile[0])
        Xdev, Ydev = read_corpus_hindi(inputfile[1])
        if state == 'init':
            divide_sentences_hindi(Xtrain, 'train')
            divide_sentences_hindi(Xdev, 'dev')
        if state == 'concatenate':
            normalised = concatenate_sentences_hindi(
                Xtrain, Ytrain, 'hing_normalised_train.txt')
            normalised_dev = concatenate_sentences_hindi(
                Xdev, Ydev, 'hing_normalised_dev.txt')
            with open('normalised_hindi_train.json', 'w') as out:
                out.write(json.dumps(normalised))
            with open('normalised_hindi_dev.json', 'w') as out:
                out.write(json.dumps(normalised_dev))

    if language == 'spanglish':
        Xtrain, Ytrain = read_corpus_spa(inputfile[0])
        Xdev, Ydev = read_corpus_spa(inputfile[1])
        if state == 'init':
            divide_sentences_es(Xtrain, 'train')
            divide_sentences_es(Xdev, 'dev')
        if state == 'concatenate':
            normalised_spa_train = concatenate_sentences_es(Xtrain, Ytrain, 'train')
            with open('normalised_spanglish_train.json', 'w') as out:
                out.write(json.dumps(normalised_spa_train))
            normalised_spa_dev = concatenate_sentences_es(Xdev, Ydev, 'dev')
            with open('normalised_spanglish_dev.json', 'w') as out:
                out.write(json.dumps(normalised_spa_dev))
    if language == 'extra':
        Xextra, Yextra = read_extra_data(inputfile[0])
        if state == 'init':
            divide_sentences_es(Xextra)
        if state == 'concatenate':
            normalised_extra = concatenate_sentences_extra(Xextra, Yextra)
            concatenate_and_add_label(normalised_extra)
            print(len(normalised_extra))
            with open('normalised_extra.json', 'w') as out:
                out.write(json.dumps(normalised_extra)) 
if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', help="input files", nargs="*", default="")
    parser.add_argument('--language', help="give the language, hinglish, spanglish or extra")
    parser.add_argument('--type', help="type, init or concatenate")
    args = parser.parse_args()
    main(args.files,args.language, args.type)
