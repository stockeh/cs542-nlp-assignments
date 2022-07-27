# CS542 Fall 2021 Programming Assignment 3
# Part-of-speech Tagging with Structured Perceptrons
# Jason Stock

import os
import numpy as np
from collections import defaultdict
import random
from random import Random

from numpy.core.fromnumeric import argmax


class POSTagger():

    def __init__(self):
        # for testing with the toy corpus from worked example
        self.tag_dict = {'nn': 0, 'vb': 1, 'dt': 2}
        self.word_dict = {'Alice': 0, 'admired': 1, 'Dorothy': 2, 'every': 3,
                          'dwarf': 4, 'cheered': 5}
        # initial tag weights [shape = (len(tag_dict),)]
        self.initial = np.array([-0.3, -0.7, 0.3])
        # tag-to-tag transition weights [shape = (len(tag_dict),len(tag_dict))]
        self.transition = np.array([[-0.7, 0.3, -0.3],
                                    [-0.3, -0.7, 0.3],
                                    [0.3, -0.3, -0.7]])
        # tag emission weights [shape = (len(word_dict),len(tag_dict))]
        self.emission = np.array([[-0.3, -0.7, 0.3],
                                  [0.3, -0.3, -0.7],
                                  [-0.3, 0.3, -0.7],
                                  [-0.7, -0.3, 0.3],
                                  [0.3, -0.7, -0.3],
                                  [-0.7, 0.3, -0.3]])
        self.unk_index = -1

    def make_dicts(self, train_set):
        '''
        Fills in self.tag_dict and self.word_dict, based on the training data.
        '''
        tag_vocabulary = set()
        word_vocabulary = set()
        # iterate over training documents
        for root, dirs, files in os.walk(train_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    # BEGIN STUDENT CODE
                    # create vocabularies of every tag and word
                    #  that exists in the training data
                    for line in f:
                        line = line.strip()  # remove empty sentence
                        if line:
                            for word in line.split():
                                partition = word.rpartition('/')
                                word_vocabulary.add(partition[0])
                                tag_vocabulary.add(partition[-1])
                    # END STUDENT CODE
        # create tag_dict and word_dict
        # if you implemented the rest of this
        #  function correctly, these should be formatted
        #  as they are above in __init__
        self.tag_dict = {v: k for k, v in enumerate(tag_vocabulary)}
        self.word_dict = {v: k for k, v in enumerate(word_vocabulary)}

    def load_data(self, data_set):
        '''
        Loads a dataset. Specifically, returns a list of sentence_ids, and
        dictionaries of tag_lists and word_lists such that:
        tag_lists[sentence_id] = list of part-of-speech tags in the sentence
        word_lists[sentence_id] = list of words in the sentence
        '''
        # doc name + ordinal number of sentence (e.g., ca010)
        sentence_ids = []
        sentences = dict()
        tag_lists = dict()
        word_lists = dict()
        # iterate over documents
        for root, dirs, files in os.walk(data_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    # be sure to split documents into sentences here
                    # BEGIN STUDENT CODE
                    # for each sentence in the document
                    # TODO: assumes sentances are new line separated. Empty line ignored.
                    i = 0
                    for line in f:
                        #  1) create a list of tags and list of words that
                        #     appear in this sentence
                        line = line.strip()  # remove empty sentence
                        if line:
                            words, tags = [], []
                            for word in line.split():
                                partition = word.rpartition('/')
                                try:
                                    wi = self.word_dict[partition[0]]
                                except KeyError:  # an unknown word
                                    wi = self.unk_index
                                try:
                                    ti = self.tag_dict[partition[-1]]
                                except KeyError:  # an unknown tag
                                    ti = self.unk_index
                                words.append(wi)
                                tags.append(ti)
                            #  2) create the sentence ID, add it to sentence_ids
                            sentence_id = f'{name}{i}'
                            sentence_ids.append(sentence_id)
                            #  3) add this sentence's tag list to tag_lists and word
                            #     list to word_lists
                            sentences[sentence_id] = line
                            word_lists[sentence_id] = words
                            tag_lists[sentence_id] = tags
                            i += 1  # only increment after valid line
                    # END STUDENT CODE
        return sentence_ids, sentences, tag_lists, word_lists

    def viterbi(self, sentence):
        '''
        Implements the Viterbi algorithm.
        Use v and backpointer to find the best_path.
        '''
        T = len(sentence)
        N = len(self.tag_dict)
        v = np.zeros((N, T))
        backpointer = np.zeros((N, T), dtype=int)
        best_path = []
        # BEGIN STUDENT CODE
        # initialization step
        #  fill out first column of viterbi trellis
        #  with initial + emission weights of the first observation
        wi = sentence[0]
        b = self.emission[wi, :] if wi != self.unk_index else 0
        v[:, 0] = self.initial + b
        # recursion step
        for t in range(1, T):
            wi = sentence[t]
            b = self.emission[wi, None, :] if wi != self.unk_index else 0
            vv = v[:, t-1, None] + self.transition + b

            #  1) fill out the t-th column of viterbi trellis
            #  with the max of the t-1-th column of trellis
            #  + transition weights to each state
            #  + emission weights of t-th observateion
            v[:, t] = np.max(vv, axis=0)
            #  2) fill out the t-th column of the backpointer trellis
            #  with the associated argmax values
            backpointer[:, t] = np.argmax(vv, axis=0)
        # termination step
        #  1) get the most likely ending state, insert it into best_path
        #  2) fill out best_path from backpointer trellis
        print(v, backpointer, sep='\n')
        best_path.append(np.argmax(v[:, -1]))
        end = best_path[-1]
        for i in reversed(range(1, T)):
            best_path.append(backpointer[end, i])
            end = best_path[-1]
        best_path = list(reversed(best_path))
        # END STUDENT CODE
        return best_path

    def train(self, train_set, dummy_data=None):
        '''
        Trains a structured perceptron part-of-speech tagger on a training set.
        '''
        self.make_dicts(train_set)
        sentence_ids, sentences, \
            tag_lists, word_lists = self.load_data(train_set)
        if dummy_data is None:  # for automated testing: DO NOT CHANGE!!
            Random(1).shuffle(sentence_ids)
            self.initial = np.zeros(len(self.tag_dict))
            self.transition = np.zeros(
                (len(self.tag_dict), len(self.tag_dict)))
            self.emission = np.zeros((len(self.word_dict), len(self.tag_dict)))
        else:
            sentence_ids = dummy_data[0]
            sentences = dummy_data[1]
            tag_lists = dummy_data[2]
            word_lists = dummy_data[3]
        for i, sentence_id in enumerate(sentence_ids):
            # BEGIN STUDENT CODE
            # get the word sequence for this sentence and the correct tag sequence
            words = word_lists[sentence_id]
            tags = tag_lists[sentence_id]
            # use viterbi to predict
            best_path = self.viterbi(words)
            # if mistake
            if tags != best_path:
                eta = 1.0
                #  promote weights that appear in correct sequence
                #  demote weights that appear in (incorrect) predicted sequence
                self.initial[tags[0]] += eta
                self.initial[best_path[0]] -= eta
                for best, tag, word in zip(best_path, tags, words):
                    self.emission[word, tag] += eta
                    self.emission[word, best] -= eta
                # each row represents a previous tag,
                # while each column represents a current tag
                for j in range(1, len(tags)):
                    self.transition[tags[j-1], tags[j]] += eta
                    self.transition[best_path[j-1], best_path[j]] -= eta
            # END STUDENT CODE
            if (i + 1) % 1000 == 0 or i + 1 == len(sentence_ids):
                print(i + 1, 'training sentences tagged')

    def test(self, dev_set, dummy_data=None):
        '''
        Tests the tagger on a development or test set.
        Returns a dictionary of sentence_ids mapped to their correct and predicted
        sequences of part-of-speech tags such that:
        results[sentence_id]['correct'] = correct sequence of tags
        results[sentence_id]['predicted'] = predicted sequence of tags
        '''
        results = defaultdict(dict)
        sentence_ids, sentences, tag_lists, word_lists = self.load_data(
            dev_set)
        if dummy_data is not None:  # for automated testing: DO NOT CHANGE!!
            sentence_ids = dummy_data[0]
            sentences = dummy_data[1]
            tag_lists = dummy_data[2]
            word_lists = dummy_data[3]
        for i, sentence_id in enumerate(sentence_ids):
            # BEGIN STUDENT CODE
            results[sentence_id]['correct'] = tag_lists[sentence_id]
            results[sentence_id]['predicted'] = self.viterbi(
                word_lists[sentence_id])
            # END STUDENT CODE
            if (i + 1) % 1000 == 0 or i + 1 == len(sentence_ids):
                print(i + 1, 'testing sentences tagged')
        return sentences, results

    def evaluate(self, sentences, results, dummy_data=False):
        '''
        Given results, calculates overall accuracy.
        This evaluate function calculates accuracy ONLY,
        no precision or recall calculations are required.
        '''
        if not dummy_data:
            self.sample_results(sentences, results)
        accuracy = 0.0
        # BEGIN STUDENT CODE
        T = [r['correct'] for r in results.values()]
        Y = [r['predicted'] for r in results.values()]
        total_words = 0.0
        total_correct_words = 0.0
        for t, y in zip(T, Y):
            total_words += len(t)
            for i in range(len(t)):
                if t[i] == y[i]:
                    total_correct_words += 1
        accuracy = total_correct_words / total_words
        # END STUDENT CODE
        return accuracy

    def sample_results(self, sentences, results, size=2):
        '''
        Prints out some sample results, with original sentence,
        correct tag sequence, and predicted tag sequence.
        This is just to view some results in an interpretable format.
        You do not need to do anything in this function.
        '''
        print('\nSample results')
        results_sample = [random.choice(list(results)) for i in range(size)]
        inv_tag_dict = {v: k for k, v in self.tag_dict.items()}
        for sentence_id in results_sample:
            length = len(results[sentence_id]['correct'])
            correct_tags = [inv_tag_dict[results[sentence_id]
                                         ['correct'][i]] for i in range(length)]
            predicted_tags = [inv_tag_dict[results[sentence_id]
                                           ['predicted'][i]] for i in range(length)]
            print(sentence_id,
                  sentences[sentence_id],
                  'Correct:\t', correct_tags,
                  '\n Predicted:\t', predicted_tags, '\n')


if __name__ == '__main__':
    pos = POSTagger()
    # make sure these point to the right directories
    # pos.train('data_small/train')  # train: toy data
    # pos.train('brown_news/train')  # train: news data only
    pos.train('brown/train')  # train: full data
    # sentences, results = pos.test('data_small/test')  # test: toy data
    # sentences, results = pos.test('brown_news/dev')  # test: news data only
    sentences, results = pos.test('brown/dev')  # test: full data
    print('\nAccuracy:', pos.evaluate(sentences, results))
