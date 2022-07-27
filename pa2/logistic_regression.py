# CS542 Spring 2021 Programming Assignment 2
# Logistic Regression Classifier
# Jason Stock

import os
import time
import csv
import copy
import numpy as np
import itertools
from collections import defaultdict
from math import ceil
from random import Random


def sigma(z):
    '''Computes the logistic function.'''
    return 1 / (1 + np.exp(-z))


def hyperparam_search():
    n_epochs = [2, 8, 16, 32, 64, 128]
    batch_sizes = [2, 4, 8, 16, 32]
    etas = [0.0001, 0.001, 0.01, 0.1]
    featurize_methods = ['count', 'binary']
    select_features_methods = ['all', 'lexicon', 'length',
                               '2gram_2_200', '3gram_2_300']
    standardizes = [True, False]
    stems = [True, False]
    items = list(itertools.product(n_epochs, batch_sizes, etas,
                                   featurize_methods, select_features_methods, standardizes, stems))
    items += list(itertools.product(n_epochs, batch_sizes,
                  etas, ['stats'], [None], standardizes, stems))
    results = []
    for i, (epochs, batch_size, eta, featurize_method, select_features_method, standardize, stem) in enumerate(items):
        print(f'{i+1}/{len(items)}')
        try:
            lr = LogisticRegression(featurize_method=featurize_method,
                                    select_features_method=select_features_method,
                                    standardize=standardize,
                                    stem=stem)
            lr.train('movie_reviews/train', batch_size=batch_size,
                     n_epochs=epochs, eta=eta, verbose=False)
            accuracy = lr.evaluate(lr.test('movie_reviews/dev'), verbose=False)
            results.append({'accuracy': accuracy, 'n_epoch': epochs, 'batch_size': batch_size, 'eta': eta,
                            'featurize_method': featurize_method, 'select_features_method': select_features_method,
                            'standardize': standardize, 'stem': stem, 'n_features': lr.n_features,
                            'training_time': lr.training_time, 'error_trace': lr.error_trace})
        except Exception as e:
            print(e)
            continue  # exception caught during training
    keys = results[0].keys()

    with open('results.csv', 'w', newline='') as of:
        writer = csv.DictWriter(of, keys)
        writer.writeheader()
        writer.writerows(results)


class Stemmer():
    '''Modified Porter Word Stemming Algorithm.
    Code adopted from Jason Stock and converted from C++ to Python.
    (https://github.com/stockeh/search-engine/blob/master/src/stemming.cpp)
    '''

    def __init__(self, exception_path='Exceptions.txt', inputdir=None) -> None:
        self.exception_path = exception_path
        if os.path.isfile(exception_path):
            with open(exception_path, 'r') as f:  # source: destination pairs
                self.exceptions = {k: v for k, v in
                                   [l.rstrip().split() for l in f.readlines()]}
        else:
            print(
                'No Exception file found. Stemming may not be consistent, e.g. skies -> ski != sky')
            self.exceptions = {}

        self.inputdir = inputdir
        if inputdir is not None:
            self.documents = {}
            for root, _, files in os.walk(inputdir):
                for name in files:
                    with open(os.path.join(root, name), 'r') as f:
                        self.documents[name] = f.read().split()
        self.vowels = ['a', 'e', 'i', 'o', 'u']

    def iterate_documents(self) -> dict:
        assert self.inputdir is not None, 'What documents are being iterated? __init__(..., inputdir)'
        output = {}
        for file, words in self.documents.items():
            new_words = []
            for word in words:
                new_words.append(self.stem(word))
            output[file] = new_words
        return output

    def _suffix(self, word, suffix) -> bool:
        return len(word) >= len(suffix) and word[-len(suffix):] == suffix

    def _yvowel(self, word, index) -> bool:
        if word[index+1] != 'y' and index == 0:
            return True
        c = word[index]
        if c == 'y':
            return self._isvowel(word, index-1)
        return c not in self.vowels

    def _isvowel(self, word, index) -> bool:
        if index == -1:
            return True
        c = word[index]
        if c == 'y' and index == 0:
            return False
        if c == 'y':  # recursivly check if it follows a vowel or not
            # true if does not follow vowel
            return self._yvowel(word, index-1)
        return c in self.vowels

    def _region(self, word) -> str:
        """Substring that follows the first consonate that follows a vowel.
        Region1 may be empty (often for short words).

        e.g., region2(definition) = region1(inition) = ition"""
        start = 0
        is_y = False
        if len(word) > 2:
            if word[0] == 'y' and self.region1:
                is_y = True
            for index in range(1, len(word)):
                if (not self._isvowel(word, index) and self._isvowel(word, index - 1)) or is_y:
                    start = index + 1
                    break
        return word[start:] if start > 0 else ''

    def _short_syllable(self, word) -> bool:
        length = len(word)
        if length > 2:  # ends with non-vowel followed by vowel followed by non-vowel that is not 'w' | 'x' | 'y'
            if not self._isvowel(word, length-3) and self._isvowel(word, length-2) and \
                    not self._isvowel(word, length-1):
                return False if word[-1] in ['w', 'x', 'y'] else True
            else:
                return False  # it just isn't a short syllable
        # True if two char string and vowel followed by non-vowel
        return self._isvowel(word, length-2) and \
            not self._isvowel(word, length-1) if length == 2 else False

    def _isshort(self, word) -> bool:
        return self._short_syllable(word) and not self._region(word)

    def _is_li_ending(self, c) -> bool:
        return c in ['c', 'd', 'e', 'g', 'h', 'k', 'm', 'n', 'r', 't']

    def _step_three_check(self, word, suffix) -> str:
        pre = word[:-len(suffix)]
        contains_vowel = False

        for index in range(0, len(pre)):
            if self._isvowel(pre, index):
                contains_vowel = True
                break
        if contains_vowel:
            if self._suffix(pre, 'at') or self._suffix(pre, 'bl') or self._suffix(pre, 'iz'):
                word = pre + 'e'
            elif self._suffix(pre, 'bb') or self._suffix(pre, 'dd') or self._suffix(pre, 'ff') or \
                    self._suffix(pre, 'gg') or self._suffix(pre, 'mm') or self._suffix(pre, 'nn') or \
                    self._suffix(pre, 'pp') or self._suffix(pre, 'rr') or self._suffix(pre, 'tt'):
                word = pre[:-1]
            elif self._isshort(pre):
                word = pre + 'e'
            else:
                word = pre
        return word

    def _step_one(self, word) -> str:
        if word[0] == '\'':
            word = word[1:]
        if self._suffix(word, "\'s\'"):
            word = word[:-3]
        elif self._suffix(word, "\'s"):
            word = word[:-2]
        elif self._suffix(word, "\'"):
            word = word[:-1]
        return word

    def _step_two(self, word) -> str:
        if self._suffix(word, 'sses'):
            word = word[:-2]
        elif self._suffix(word, 'ied') or self._suffix(word, 'ies'):
            if len(word) > 4:
                word = word[:-2]
            else:
                word = word[:-1]
        elif len(word) > 1:
            if self._suffix(word, 's'):
                contains_vowel = False
                for index in range(0, len(word) - 2):
                    if self._isvowel(word, index):
                        contains_vowel = True
                        break
                if contains_vowel:
                    word = word[:-1]
        return word

    def _step_three(self, word) -> str:
        if self._suffix(word, 'eedly'):
            self.region1 = self._region(word)
            if self.region1 == 'eedly':
                word = word[:-3]
            self.region1 = ''
        elif self._suffix(word, 'ingly'):
            word = self._step_three_check(word, 'ingly')
        elif self._suffix(word, 'edly'):
            word = self._step_three_check(word, 'edly')
        elif self._suffix(word, 'eed'):
            self.region1 = self._region(word)
            if self.region1 == 'eed':
                word = word[:-1]
            self.region1 = ''
        elif self._suffix(word, 'ing'):
            word = self._step_three_check(word, 'ing')
        elif self._suffix(word, 'ed'):
            word = self._step_three_check(word, 'ed')
        self.region1 = ''
        return word

    def _step_four(self, word) -> str:
        if self._suffix(word, 'y') and len(word) > 2 and not self._isvowel(word, len(word)-2):
            word = word[:-1] + 'i'
        return word

    def _seven_five(self, word):
        if self._suffix(word, 'ization') or self._suffix(word, 'ational'):
            return word[:-5] + 'e', True
        elif self._suffix(word, 'fulness') or self._suffix(word, 'ousness') or \
                self._suffix(word, 'iveness'):
            return word[:-4], True
        return word, False

    def _six_five(self, word):
        if self._suffix(word, 'tional') or self._suffix(word, 'lessli'):
            return word[:-2], True
        elif self._suffix(word, 'biliti'):
            return word[:-5] + 'le', True
        return word, False

    def _five_five(self, word):
        if self._suffix(word, 'entli') or self._suffix(word, 'ousli') or self._suffix(word, 'fulli'):
            return word[:-2], True
        elif self._suffix(word, 'ation') or self._suffix(word, 'iviti'):
            return word[:-3] + 'e', True
        elif self._suffix(word, 'alism') or self._suffix(word, 'aliti'):
            return word[:-3], True
        return word, False

    def _four_five(self, word):
        if self._suffix(word, 'enci') or self._suffix(word, 'anci') or self._suffix(word, 'abli'):
            return word[:-1] + 'e', True
        elif self._suffix(word, 'ator'):
            return word[:-2] + 'e', True
        elif self._suffix(word, 'alli'):
            return word[:-2], True
        elif self._suffix(word, 'izer'):
            return word[:-1], True
        return word, False

    def _three_five(self, word):
        if self._suffix(word, 'bli'):
            return word[:-1] + 'e', True
        elif len(word) > 3:  # suffix preceded by l
            if self._suffix(word, 'logi'):
                return word[:-1], True
        return word, False

    def _two_five(self, word):
        if self._suffix(word, 'li'):
            if len(word) > 2:
                if self._is_li_ending(word[-3]):
                    return word[:-2], True
        return word, False

    def _step_five(self, word) -> str:
        c = word[-1]
        length = len(word)
        if c in ['n', 'l', 's'] and length >= 7:
            word, stop = self._seven_five(word)
            if stop:
                return word
        if c in ['i', 'l'] and length >= 6:
            word, stop = self._six_five(word)
            if stop:
                return word
        if c in ['n', 'i', 'm'] and length >= 5:
            word, stop = self._five_five(word)
            if stop:
                return word
        if c in ['i', 'r'] and length >= 4:
            word, stop = self._four_five(word)
            if stop:
                return word
        if c in ['i'] and length >= 3:
            word, stop = self._three_five(word)
            if stop:
                return word
        word, _ = self._two_five(word)
        return word

    def _step_six(self, word) -> str:
        self.region1 = self._region(word)
        if self._suffix(word, 'ational'):
            if 'ational' in self.region1:
                word = word[:-7] + 'ate'
        elif self._suffix(word, 'tional'):
            if 'tional' in self.region1:
                word = word[:-2]
        elif self._suffix(word, 'alize'):
            if 'alize' in self.region1:
                word = word[:-3]
        elif self._suffix(word, 'ative'):
            if 'ative' in self._region(self.region1):
                word = word[:-5]
        elif self._suffix(word, 'icate'):
            if 'icate' in self.region1:
                word = word[:-3]
        elif self._suffix(word, 'iciti'):
            if 'iciti' in self.region1:
                word = word[:-3]
        elif self._suffix(word, 'ical'):
            if 'ical' in self.region1:
                word = word[:-2]
        elif self._suffix(word, 'ical'):
            if 'ical' in self.region1:
                word = word[:-2]
        elif self._suffix(word, 'ness'):
            if 'ness' in self.region1:
                word = word[:-4]
        elif self._suffix(word, 'ful'):
            if 'ful' in self.region1:
                word = word[:-3]
        self.region1 = ''
        return word

    def _five_seven(self, word, region2):
        if self._suffix(word, 'ement'):
            if 'ement' in region2:
                return word[:-5], True
        return word, False

    def _four_seven(self, word, region2):
        if self._suffix(word, 'ance'):
            if 'ance' in region2:
                return word[:-4], True
        elif self._suffix(word, 'ence'):
            if 'ence' in region2:
                return word[:-4], True
        elif self._suffix(word, 'able'):
            if 'able' in region2:
                return word[:-4], True
        elif self._suffix(word, 'ible'):
            if 'ible' in region2:
                return word[:-4], True
        elif self._suffix(word, 'ment'):
            if 'ment' in region2:
                return word[:-4], True
        return word, False

    def _three_seven(self, word, region2):
        if self._suffix(word, 'ant'):
            if 'ant' in region2:
                return word[:-3], True
        elif self._suffix(word, 'ent'):
            if 'ent' in region2:
                return word[:-3], True
        elif self._suffix(word, 'ism'):
            if 'ism' in region2:
                return word[:-3], True
        elif self._suffix(word, 'ate'):
            if 'ate' in region2:
                return word[:-3], True
        elif self._suffix(word, 'iti'):
            if 'iti' in region2:
                return word[:-3], True
        elif self._suffix(word, 'ous'):
            if 'ous' in region2:
                return word[:-3], True
        elif self._suffix(word, 'ive'):
            if 'ive' in region2:
                return word[:-3], True
        elif self._suffix(word, 'ize'):
            if 'ize' in region2:
                return word[:-3], True
        elif self._suffix(word, 'ion'):
            if 'ion' in region2 and len(word) > 3:
                if word[-4] in ['s', 't']:
                    return word[:-3], True
        return word, False

    def _two_seven(self, word, region2):
        if self._suffix(word, 'al'):
            if 'al' in region2:
                return word[:-2], True
        elif self._suffix(word, 'er'):
            if 'er' in region2:
                return word[:-2], True
        elif self._suffix(word, 'ic'):
            if 'ic' in region2:
                return word[:-2], True
        return word, False

    def _step_seven(self, word) -> str:
        self.region1 = self._region(word)
        self.region2 = self._region(self.region1)
        c = word[-1]
        length = len(word)
        if c == 't' and length >= 5:
            word, stop = self._five_seven(word, self.region2)
            if stop:
                return word
        if c in ['e', 't'] and length >= 4:
            word, stop = self._four_seven(word, self.region2)
            if stop:
                return word
        if c in ['e', 't', 'm', 'i', 's', 'n'] and length >= 3:
            word, stop = self._three_seven(word, self.region2)
            if stop:
                return word
        word, _ = self._three_seven(word, self.region2)
        self.region1 = ''
        self.region2 = ''
        return word

    def _step_eight(self, word) -> str:
        self.region1 = self._region(word)
        self.region2 = self._region(self.region1)
        if self._suffix(word, 'e'):
            if 'e' in self.region2:
                word = word[:-1]
            elif 'e' in self.region1:
                if not self._short_syllable(word[:-1]):
                    word = word[:-1]
        elif self._suffix(word, 'l'):
            if 'l' in self.region2 and self._suffix(word, 'll'):
                word = word[:-1]
        self.region1 = ''
        self.region2 = ''
        return word

    def stem(self, word, verbose=False):
        isupper = word[0].isupper()
        lenof2 = len(word) > 2
        if word in self.exceptions and not isupper and lenof2:  # replace word
            return self.exceptions[word]
        elif isupper or not lenof2:
            return word

        if verbose:
            print(word, end=', ')
        # TODO: do regions1&2 update after each step or should it only be computed first?
        self.region1, self.region2 = '', ''
        word = self._step_one(word)
        word = self._step_two(word)
        word = self._step_three(word)
        word = self._step_four(word)
        if len(word) > 2:
            word = self._step_five(word)
        word = self._step_six(word)
        if len(word) > 2:
            word = self._step_seven(word)
        word = self._step_eight(word)
        if verbose:
            print(word)
        return word


class LogisticRegression():

    def __init__(self, n_features=-1, featurize_method='count', select_features_method='default', standardize=False, stem=False):
        self.class_dict = {'neg': 0, 'pos': 1}
        # self.class_dict = {'action': 0, 'comedy': 1}

        if featurize_method == 'stats':
            n_features = 8
            select_features_method = None

        self.featurize_method = featurize_method
        self.select_features_method = select_features_method
        self.standardize = standardize
        self.stem = stem
        if stem:
            self.stemmer = Stemmer(exception_path='Exceptions.txt')

        self.n = None  # ngram degree

        self.feature_dict = None
        if n_features == -1:
            self.n_features = None
            self.theta = None
        else:
            self.n_features = n_features
            self.theta = np.zeros(self.n_features + 1)

        # training standardization vars
        self.Xmeans = None
        self.Xstds = None

        # additional metadata
        self.error_trace = []
        self.training_time = 0
        self.n_epochs = 0
        self.batch_size = 0
        self.eta = 0

        # potential adpositions, determiners, and pronouns
        self.stop = set(['the', 'a', 'an', 'is', 'and', 'this', 'that', 'these', 'those', 'some', 'most', 'my',
                         'your', 'he', 'his', 'her', 'she', 'its', 'our', 'their', 'which', 'on', 'of', 'at', 'by',
                         'into', 'through', 'till', 'to', 'toward', 'with', '.', ',', '?', ')'])
        self.positive = set(['best', 'yes', 'good', 'safe', 'again', 'recommend', 'actor', 'felt', 'laugh', 'friend',
                             'postive', 'would', 'photography', 'visuals', 'soundtrack', 'home', 'friend\'s', 'wow', 'cozy',
                             'wonderful', 'amazing', 'brilliant', 'theaters', 'cool', 'kids', 'favorite', 'suspenseful',
                             'amazed', 'candy', 'warm', 'happy', 'birthday', 'baby', 'puppy', 'however'])
        self.negative = set(['bad', 'no', 'watch', 'wouldn\'t', 'scared', 'sad', 'mad', 'cried', 'out', 'racist',
                             'unbelievable',  'over', 'better', 'couldn\'t',  'stop', 'negative', 'never', 'crappy',
                             'disgusting', 'ugh', 'feeling', 'horrific', 'slow', 'least', 'waiting', 'lame', 'waste',
                             'asleep', 'sleep', 'money', 'loud', 'crying', 'gruesome', 'funnier', 'blood'])
        # might remove some words, e.g. friend's -> friend
        if self.stem:
            for const in [self.stop, self.positive, self.negative]:
                tmp = list(const)
                const.clear()
                for w in tmp:
                    const.add(self.stemmer.stem(w))

    def __repr__(self):
        s = f'{self.featurize_method=}\n{self.select_features_method=}\n{self.standardize=}\n'
        s += f'{self.stem=}\n{self.n_features=}\n{self.training_time=:.4f}\n{self.n_epochs=}\n'
        s += f'{self.batch_size=}\n{self.eta=}'
        return s

    def load_data(self, data_set):
        '''Loads a dataset. Specifically, returns a list of filenames, and dictionaries
        of classes and documents such that:
        classes[filename] = class of the document
        documents[filename] = feature vector for the document (use self.featurize)
        '''
        # start_t = time.time()
        vs = dict()  # {class: [{document: list(words)}, ndocs]}
        # iterate over documents
        for root, dirs, files in os.walk(data_set):
            try:
                c = self.class_dict[root.split('/')[-1]]
                vs[c] = [dict(), 0]
            except KeyError:
                continue
            for name in files:
                with open(os.path.join(root, name)) as f:
                    # read all words to list
                    vs[c][0][name] = [self.stemmer.stem(w) for w in f.read().split()] if self.stem \
                        else f.read().split()
                vs[c][1] += 1  # increment num docs for class c

        # initialize
        if self.n_features is None:
            self.feature_dict = self.select_features(vs)
            self.n_features = len(self.feature_dict)
            self.theta = np.zeros(self.n_features + 1)  # weights (and bias)

        filenames = []
        classes = dict()
        documents = dict()
        for c in sorted(vs.keys()):
            for name, words in vs[c][0].items():
                filenames.append(name)
                classes[name] = c
                documents[name] = self.featurize(words)

        if self.standardize:
            if self.Xmeans is None:  # use training samples to standardize
                self._setup_standardize(np.asarray(
                    [v for v in documents.values()]))
            documents = {k: self._standardizeX([v])[0]
                         for k, v in documents.items()}
        # print(f'Finished Loading Data in {time.time() - start_t:.3f} s')
        return filenames, classes, documents

    def _standardizeX(self, X):
        result = (X - self.Xmeans) / self.XstdsFixed
        result[:, self.Xconstant] = 0.0
        result[:, -1] = 1.0  # keep bias = 1
        return result

    def _unstandardizeX(self, Xs):
        return self.Xstds * Xs + self.Xmeans

    def _setup_standardize(self, X):
        self.Xmeans = X.mean(axis=0)
        self.Xstds = X.std(axis=0)
        self.Xconstant = self.Xstds == 0
        self.XstdsFixed = copy.copy(self.Xstds)
        self.XstdsFixed[self.Xconstant] = 1

    def featurize(self, document):
        '''Given a document (as a list of words), returns a feature vector.
        Note that the last element of the vector, corresponding to the bias, is a
        "dummy feature" with value 1.
        '''
        vector = np.zeros(self.n_features + 1)
        if self.featurize_method in ['count', 'binary']:
            n = 1 if self.n is None else self.n
            for i in range(n, len(document)+1):
                w = tuple(document[i-n:i]) if n > 1 else document[i-n]
                if w in self.feature_dict:
                    if self.featurize_method == 'count':
                        vector[self.feature_dict[w]] += 1
                    else:
                        vector[self.feature_dict[w]] = 1
        elif self.featurize_method == 'stats':
            # this has nothing to do with select_features
            count = 0
            unique = set()
            characters = []
            ending_punc = set(['.', '?', '!', ';'])
            for w in document:
                if w in self.stop:
                    vector[0] += 1  # total stop words
                elif w[0] in ending_punc:
                    vector[1] += 1  # total sentences
                elif w in self.positive:
                    vector[2] += 1  # total positive words
                elif w in self.negative:
                    vector[3] += 1  # total negative words
                count += 1
                unique.add(w)
                characters.append(len(w))

            vector[4] = len(unique) / count
            vector[5] = np.mean(characters)
            vector[6] = np.std(characters)
            vector[7] = max(characters)
        else:
            raise NotImplementedError

        vector[-1] = 1
        return vector

    def select_features(self, vs):
        '''vs = {class: [{document: list(words)}, ndocs]}'''
        if self.select_features_method == 'lexicon':
            # valid set of features to use. subsiquent docs only check if these are contained.
            features_to_use = list(self.positive.union(self.negative))
            return dict([(k, i) for i, k in enumerate(features_to_use)])
        if 'gram' in self.select_features_method:
            # create ngram feature candidates whose occurances are above/below a threshold
            # ngram_lo_hi, e.g. 2gram, 3gram_3, 2gram_2_300, etc.
            try:
                # save self.n for featurize function
                self.n = int(self.select_features_method[0])
                parts = self.select_features_method.split('_')
                lo = 0 if len(parts) == 1 else int(parts[1])
                hi = float('inf') if len(parts) in [1, 2] else int(parts[2])
            except:
                raise ValueError(
                    f'{self.select_features_method} should have the form \'ngram_lo_hi\' : n is degree')

            ngrams = defaultdict(int)
            for c in sorted(vs.keys()):
                for _, words in vs[c][0].items():
                    for i in range(self.n, len(words)+1):
                        gram = tuple(words[i-self.n:i])
                        # only add feature candidates if ngram is NOT made up from stop words
                        if len([v for v in gram if v in self.stop]) < self.n:
                            ngrams[gram] += 1
            # print('counts:', {k: v for k, v in sorted(ngrams.items(), key=lambda item: item[1]) if v > 1})
            # only keep ngrams that occur between lo and hi
            return dict([(k, i) for i, (k, _) in enumerate(filter(
                lambda item: item[1] > lo and item[1] < hi, ngrams.items()))])

        # np array of unique words across all documents and classes
        words = np.array(list(set().union(*[doc for v in vs.values()
                                            for doc in v[0].values()])))
        if len(words) < 5 or self.select_features_method == 'all':
            # add all the words if the vocabulary is small or if specified
            return dict([(k, i) for i, k in enumerate(words)])
        elif self.select_features_method == 'length':
            # add words where their character length are greater than `threshold`
            threshold_min = 3
            threshold_max = 6
            lengths = np.asarray([len(w) for w in words])
            filtered = words[(lengths >= threshold_min) &
                             (lengths <= threshold_max)]
            return dict([(k, i) for i, k in enumerate(filtered)])
        elif self.select_features_method == 'default':
            return {'fast': 0, 'couple': 1, 'shoot': 2, 'fly': 3}
        else:
            raise NotImplementedError

    def train(self, train_set, batch_size=3, n_epochs=1, eta=0.1, verbose=False):
        '''Trains a logistic regression classifier on a training set.'''
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.eta = eta
        start_t = time.time()
        filenames, classes, documents = self.load_data(train_set)
        filenames = sorted(filenames)
        n_minibatches = ceil(len(filenames) / batch_size)
        epsilon = 1e-9
        for epoch in range(n_epochs):
            if verbose:
                print(f'Epoch {epoch+1}/{n_epochs}')
            loss = 0
            for i in range(n_minibatches):
                minibatch = filenames[i * batch_size: (i + 1) * batch_size]
                X = np.array([documents[d] for d in minibatch])
                T = np.array([classes[d] for d in minibatch])
                Y = sigma(np.dot(X, self.theta))
                loss += np.sum(-(T * np.log(Y + epsilon) + (1 - T) *
                                 np.log(1 - Y + epsilon)))
                self.theta -= eta * (np.dot(X.T, Y - T) / X.shape[0])
            loss /= len(filenames)
            self.error_trace.append(np.exp(-loss))
            if verbose:
                print(
                    f'Average Train Loss: {loss:.3f}, Likelihood {self.error_trace[-1]:.3f}')
            Random(epoch).shuffle(filenames)
        self.training_time = time.time() - start_t

    def test(self, dev_set):
        '''Tests the classifier on a development or test set.
        Returns a dictionary of filenames mapped to their correct and predicted
        classes such that:
        results[filename]['correct'] = correct class
        results[filename]['predicted'] = predicted class
        '''
        assert self.theta is not None, 'Must train features before testing.'
        results = defaultdict(dict)
        filenames, classes, documents = self.load_data(dev_set)
        for name in filenames:
            # get most likely class (recall that P(y=1|x) = y_hat)
            results[name]['correct'] = classes[name]
            results[name]['predicted'] = 1 if sigma(
                np.dot(np.array(documents[name]), self.theta)) > 0.5 else 0
        return results

    def print_metrics(self, confmat, precision, recall, f1, accuracy):
        # Print Classes
        wrap = '-'*40
        print(wrap)
        print('Classes:', ', '.join(f'{k}: {v}' for k,
              v in self.class_dict.items()), end='\n\n')
        # Print Confusion Matrix
        classes = list(self.class_dict.values())
        spacing = len(str(np.max(confmat)))
        class_spacing = len(str(np.max(classes)))+1
        top = ' '*(class_spacing) + ''.join(' {i: < {spacing}}'.format(
            i=i, spacing=str(spacing)) for i in classes)
        t = [f'{classes[j]} |' + ''.join(' {i:<{spacing}}'.format(
            i=i, spacing=str(spacing)) for i in row) for j, row in enumerate(confmat)]
        hdr = ' '*class_spacing + '-'*(len(t[0]) - class_spacing)
        print('Confusion Matrix:', top, hdr, '\n'.join(t), sep='\n')
        # All-Class Metrics
        labels = ['Precision', 'Recall', 'F1']
        precision = np.append(precision, precision.mean())
        recall = np.append(recall, recall.mean())
        f1 = np.append(f1, 2*precision.mean()*recall.mean() /
                       (precision.mean()+recall.mean()))
        # Print Metrics
        metrics = np.vstack([precision, recall, f1])
        label_spacing = max([len(l) for l in labels])+1
        metric_spacing = max([len(f'{m:.3f}') for m in metrics.flatten()])
        mean = '  mean'
        top = ' '*(label_spacing) + ''.join(' {i: < {spacing}}'.format(
            i=i, spacing=str(metric_spacing)) for i in classes) + mean
        t = ['{i:<{spacing}}|'.format(i=labels[j], spacing=str(label_spacing)) + ''.join(f' {i:.3f}' for i in row)
             for j, row in enumerate(metrics)]
        hdr = ' '*label_spacing + '-'*(len(t[0]) - label_spacing)
        print('\nMetrics:', top, hdr, '\n'.join(t), sep='\n')
        # Print Accuracy
        print(f'\nOverall Accuracy: {accuracy*100:.3f} %')
        print(wrap)

    def evaluate(self, results, verbose=True):
        '''Given results, calculates the following:
        Precision, Recall, F1 for each class
        Accuracy overall
        Also, prints evaluation metrics in readable format.
        '''
        assert self.theta is not None, 'Must train features before evaluating.'
        # you can copy and paste your code from PA1 here
        confmat = np.zeros(
            (len(self.class_dict), len(self.class_dict)), dtype=int)
        T = [r['correct'] for r in results.values()]
        Y = [r['predicted'] for r in results.values()]
        for i in range(len(T)):
            confmat[T[i], Y[i]] += 1

        precision = np.diag(confmat) / \
            np.sum(confmat, axis=0)  # tp / (tp + fp)
        recall = np.diag(confmat) / \
            np.sum(confmat, axis=1)  # tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall)  # per class
        accuracy = np.trace(confmat) / len(T)

        if verbose:
            print(self.__repr__())
            self.print_metrics(confmat, precision, recall, f1, accuracy)

        return accuracy


if __name__ == '__main__':
    if False:
        # nohup python -u logistic_regression.py > log.out 2>&1 &
        hyperparam_search()
    else:
        settings = {'n_features': -1, 'featurize_method': 'count',
                    'select_features_method': '2gram_2_200', 'standardize': False, 'stem': False}
                    
        lr = LogisticRegression(settings['n_features'], settings['featurize_method'],
                                settings['select_features_method'], settings['standardize'], settings['stem'])
        lr.train('movie_reviews/train', batch_size=4,
                 n_epochs=8, eta=0.01, verbose=True)
        results = lr.test('movie_reviews/dev')
        lr.evaluate(results, verbose=True)
