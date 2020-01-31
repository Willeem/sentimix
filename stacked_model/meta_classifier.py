#!/usr/bin/env python3
# python meta_classifier.py --files ../data/train_conll_hinglish.txt ../data/trial_conll_hinglish.txt
# python meta_classifier.py --files ../data/train_conll_spanglish.txt ../data/trial_conll_spanglish.txt --translated_files ../data/spanglish_train_translated.txt ../data/spanglish_trial_translated.txt 

# Newest data
# python meta_classifier.py --files ../data/official_train_spanglish.conll ../data/official_dev_spanglish.conll

import os
import sys

import joblib
import argparse
import pickle
import numpy as np
import pandas as pd
from mlxtend.classifier import StackingCVClassifier
from sklearn import model_selection
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_validate, train_test_split)

from ml_models import *
from rule_based import rule_based_predictor

sys.path.append("..")
from preprocessing.preprocessing_data import read_and_process
from utilities.utilities import create_confusion_matrix

def do_grid_search(X, y, pipeline, parameters, title="", start=False):
    '''
    Do 5 fold cross-validated gridsearch over certain parameters and
    print the best parameters found according to accuracy
    '''
    if not start:
        print("\n#### SKIPPING GRIDSEARCH...")
    else:
        print(f"\n#### GRIDSEARCH [{title}] ...")
        grid_search = GridSearchCV(pipeline, parameters, n_jobs=6, cv=10, scoring='f1_weighted', return_train_score=True, verbose=10)
        grid_search.fit(X, y)

        df = pd.DataFrame(grid_search.cv_results_)[['params','mean_train_score','mean_test_score']]
        print(f"\n{df}\n")

        # store results for further evaluation
        with open('grid_' + title + '_pd.pickle', 'wb') as f:
            pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)        
   
        print("Best score: {0}".format(grid_search.best_score_))  
        print("Best parameters set:")  
        best_parameters = grid_search.best_estimator_.get_params()  
        for param_name in sorted(list(parameters.keys())):  
            print("\t{0}: {1}".format(param_name, best_parameters[param_name])) 


def train(pipeline, X, y, categories, show_plots=False, show_cm=False, show_report=False, folds=10, title="title"):
    '''
    Train the classifier and evaluate the results
    '''
    print(f"\n#### TRAINING... [{title}]")
    X = np.array(X)
    y = np.array(y)
    
    try:
        print(f"Classifier used: {pipeline.named_steps['clf']}")
    except AttributeError as e:
        print(f"Using Stacking Classifier")

    if title=="StackingClassifier":
        show_cm = True
        show_report = True

    accuracy = 0
    confusion_m = np.zeros(shape=(len(categories),len(categories)))
    kf = StratifiedKFold(n_splits=folds).split(X, y)
    pred_overall = np.array([])
    y_test_overall = np.array([])
    for train_index, test_index in kf: 
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        trained  = pipeline.fit(X_train, y_train) 
        pred = pipeline.predict(X_test)
        accuracy += accuracy_score(y_test, pred)
        confusion_m = np.add(confusion_m, confusion_matrix(y_test, pred, labels=categories))
        pred_overall = np.concatenate([pred_overall, pred])
        y_test_overall = np.concatenate([y_test_overall, y_test])

    print("\nAverage accuracy: %.5f"%(accuracy/folds) + "\n")

    if show_report:
        print('Classification report\n')
        print(classification_report(y_test_overall, pred_overall, digits=2))
    if show_cm:        
        print('\nConfusion matrix\n')
        print(confusion_m)

    create_confusion_matrix(confusion_m, categories, y_lim_value=3.0, title=title, show_plots= show_plots, method="TRAINING")
   

def test(classifier, Xtest, Ytest, show_cm=False, show_plots=False, show_report=False, rule_based=False, title="title"):
    '''
    Test the classifier and evaluate the results
    '''    
    print(f"\n#### TESTING... [{title}]")

    reverse_list = ["negative", "neutral", "positive"]
    Yguess = classifier.predict(Xtest)
    Ytest = [reverse_list[sent] for sent in Ytest]
    Yguess = [reverse_list[sent] for sent in Yguess]
    if rule_based:
        Yguess = rule_based_predictor(Xtest, Yguess, Ytest, inspect=True)

    try:
        print(f"Classifier used: {classifier.named_steps['clf']}")
    except AttributeError as e:
        print(f"Using Stacking Classifier")

    if title=="StackingClassifier":
        show_cm = True
        show_report = True        

    confusion_m = np.zeros(shape=(len(reverse_list), len(reverse_list)))

    print(f"\naccuracy = {round(accuracy_score(Ytest, Yguess), 5)}")

    if show_report:
        print('Classification report\n')
        print(classification_report(Ytest, Yguess))

    confusion_m = np.add(confusion_m, confusion_matrix(Ytest, Yguess, labels = reverse_list))
    if show_cm:
        print('\nConfusion matrix')
        print(confusion_m)

    create_confusion_matrix(confusion_m, reverse_list, y_lim_value=3.0, title=title, show_plots=show_plots, method="TESTING")


def up_down_scale(X, Y):
    '''
    Up or down scale different sentiment categories of the data
    '''
    extra_neg_c = 0
    Xextra, Yextra = [], []
    for x, y in zip(X, Y):
        if extra_neg_c >= 200:
            break
        elif y == "negative":
            extra_neg_c += 1
            Xextra.append(x)
            Yextra.append(y)

    X = X + Xextra
    Y = Y + Yextra

    extra_neu_c = 0
    Xextra, Yextra = [], []
    for x, y in zip(X, Y):
        if extra_neu_c >= 250:
            break
        elif y == "neutral":
            extra_neu_c += 1
            Xextra.append(x)
            Yextra.append(y)

    X = X + Xextra
    Y = Y + Yextra

    less_pos_c = 0
    newX, newY = [], []
    for x, y in zip(X, Y):
        if y == "positive" and less_pos_c < 1250:            
            less_pos_c += 1
        else:
            newX.append(x)
            newY.append(y)
    
    X = newX
    Y = newY
    return X, Y


def main(argv):
    parser = argparse.ArgumentParser(description='Control everything')
    parser.add_argument('--files', nargs="+")
    parser.add_argument('--translated_files', nargs="*", default="")
    parser.add_argument('--model', help="Please provide a .pkl model")
    parser.add_argument('--save', help="Use: --save [filename] ; Saves the model, with the given filename")
    args = parser.parse_args()

    # GridSearch Parameters
    parameters = {
        'linear': {  
            'clf__C': np.logspace(-3, 2, 6),
        },
        'rbf': {
            'clf__C': np.logspace(-3, 2, 6),
            'clf__gamma': np.logspace(-3, 2, 6),
            'clf__kernel': ['rbf']
        },
        'poly': {
            'clf__C': np.logspace(-3, 2, 6),
            'clf__gamma': np.logspace(-3, 2, 6),
            'clf_degree': np.array([0,1,2,3,4,5,6]),
            'clf__kernel': ['linear']
        },
        'PA': {
            'clf__C': np.logspace(-3, 2, 6),
            'clf__fit_intercept': [True, False], # True, False
            'clf__max_iter': np.array([1000, 2000, 3000, 5000, 7500]),
            'clf__loss': ['hinge', 'squared_hinge'],
            'clf__class_weight': ['balanced', None] # balanced, None
        },
    }
    algorithms = ['linear', 'rbf', 'poly', 'PA']

    if len(args.files) == 2:
        Xtrain, Ytrain = read_and_process(args.files[0], title="Train")
        Xtest, Ytest = read_and_process(args.files[1], title="Test")

        # Up / Down Scaling (1500 / 250 / 1000)
        scale = False
        if scale:
            Xtrain, Ytrain = up_down_scale(Xtrain, Ytrain)

        if len(args.translated_files) == 2:
            Xtrain_tr, _ = read_and_process(args.translated_files[0], title="Train translated", conll=False, Y_already=Ytrain)
            Xtrain_tr, _ = up_down_scale(Xtrain_tr, Ytrain, scale)
            Xtest_tr, _ = read_and_process(args.translated_files[1], title="Test translated", conll=False, Y_already=Ytest)  
        else:
            Xtrain_tr = [("", "", "")] * len(Xtrain)
            Xtest_tr = [("", "", "")] * len(Xtest)      
        Xtrain = [(*features, *translated) for (features, translated) in zip(Xtrain, Xtrain_tr)]
        Xtest = [(*features, *translated) for (features, translated) in zip(Xtest, Xtest_tr)]
        assert len(Xtrain) == len(Xtrain_tr) == len(Ytrain)
        assert len(Xtest) == len(Xtest_tr) == len(Ytest)            
    else:
        print("Usage: python3 meta_classifier.py --files <trainset> <testset>", file=sys.stderr)

    translation_dict = {"negative": 0, "neutral": 1, "positive": 2}
    Ytrain = np.array([translation_dict[sent] for sent in Ytrain])
    Ytest = np.array([translation_dict[sent] for sent in Ytest])

    classifier_words = model_words()
    classifier_meta = model_meta()
    classifier_translated_eng = model_translated_english()

    start_grid = False
    do_grid_search(Xtrain, Ytrain, classifier_words, parameters[algorithms[3]], title=algorithms[3], start=start_grid)
    if start_grid:
        return 0

    sclf = StackingCVClassifier(classifiers=[classifier_words],
                                use_probas=False,
                                meta_classifier=classifier_meta,
                                random_state=42)                      

    if args.model:
        the_classifier = joblib.load(args.model)
        test(the_classifier, Xtest, Ytest, title='StackingClassifier')
    else:
        for clf, label in zip([classifier_words, classifier_translated_eng, sclf], ['Words_SVM', 'Words_Translated_Eng', 'StackingClassifier']):
            if (label != "Words_SVM" and len(args.translated_files) != 2):
                continue
            train(clf, Xtrain, Ytrain, categories=[0, 1, 2], show_report=False, title=label, folds=10)
            if args.save:
                joblib.dump(clf, args.save+".pkl") 
            #TODO: Maybe add rule_based to argsparse
            test(clf, Xtest, Ytest, title=label, rule_based=False, show_report=True)

if __name__ == '__main__':
    main(sys.argv)
