# CS542 Spring 2021 Programming Assignment 1
# Naive Bayes Classifier and Evaluation
# Jason Stock

import os
import time
import numpy as np
from collections import defaultdict, Counter


class NaiveBayes():

    def __init__(self):
        # be sure to use the right class_dict for each data set
        # self.class_dict = {'action': 0, 'comedy': 1}
        self.class_dict = {'neg': 0, 'pos': 1}
        self.feature_dict = {}
        self.prior = None
        self.likelihood = None

    def train(self, path):
        '''Trains a multinomial Naive Bayes classifier on a training set.
        Specifically, fills in self.prior and self.likelihood such that:
        self.prior[class] = log(P(class))
        self.likelihood[class][feature] = log(P(feature|class))
        '''
        vs = dict()  # {class: [{word: count}, ndocs]}
        for root, dirs, files in os.walk(path):
            try:
                c = self.class_dict[root.split('/')[-1]]
                vs[c] = [defaultdict(int), 0]  # [wordcounts, ndocs]
            except KeyError:
                continue
            for name in files:  # collect class counts and feature counts
                with open(os.path.join(root, name)) as f:
                    for line in f:
                        for word in line.split():
                            vs[c][0][word] += 1
                        # print(c, [word for word in line.split()])
                    vs[c][1] += 1
        V = len(set().union(*(v[0].keys() for v in vs.values())))
        Ndoc = sum([v[1] for v in vs.values()])
        self.feature_dict = self.select_features(vs)
        self.prior = np.zeros(len(vs))
        self.likelihood = np.zeros((len(vs), len(self.feature_dict)))
        # normalize counts to probabilities, and take logs
        for c in sorted(vs.keys()):
            self.prior[c] = np.log(vs[c][1] / Ndoc)
            sigmawc = sum(vs[c][0].values())
            for k, v in self.feature_dict.items():
                self.likelihood[c, v] = np.log(
                    (vs[c][0][k] + 1) / (sigmawc + V))

    def test(self, path):
        '''Tests the classifier on a development or test set.
        Returns a dictionary of filenames mapped to their correct and predicted
        classes such that:
        results[filename]['correct'] = correct class
        results[filename]['predicted'] = predicted class
        '''
        results = defaultdict(dict)
        # iterate over testing documents
        for root, dirs, files in os.walk(path):
            try:
                T = self.class_dict[root.split('/')[-1]]
            except KeyError:
                continue
            for name in files:
                with open(os.path.join(root, name)) as f:
                    features = np.zeros(len(self.feature_dict))
                    for l in f:
                        for w in l.split():
                            try:
                                features[self.feature_dict[w]] += 1
                            except KeyError:
                                continue
                    results[name]['correct'] = T
                    results[name]['predicted'] = np.argmax(
                        self.prior + self.likelihood @ features)
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

    def evaluate(self, results):
        '''Given results, calculates the following:
        Precision, Recall, F1 for each class, accuracy overall
        Also, prints evaluation metrics in readable format.
        '''
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

        self.print_metrics(confmat, precision, recall, f1, accuracy)

    def select_features(self, vs, method='threshold'):
        '''Performs feature selection.
        Returns a dictionary of features.
        '''
        classes = list(self.class_dict.values())
        words = Counter(vs[classes[0]][0])
        for c in classes[1:]:  # counts of all words across all classes
            words = words + Counter(vs[classes[c]][0])
        words = np.asarray([(k, int(v))
                           for k, v in words.items()], dtype=object)  # (word, count)
        if len(words) < 5 or method == 'all':
            # add all the words if the vocabulary is small or if specified
            return dict([(k, i) for i, k in enumerate(words[:, 0])])
        elif method == 'manual':
            valid = ['good', 'bad', 'no', 'watch', 'would', 'wouldn\'t', 'happy', 'scared', 'sad',
                     'mad', 'cried', 'again', 'out', 'unbelievable', 'wow', 'ugh', 'yes', 'okay',
                     'recommend', 'friends', 'over', 'better', 'best', 'amazing', 'couldn\'t',
                     'theaters', 'cozy', 'home', 'safe', 'laugh', 'laughed', 'friend', 'felt', 'feeling',
                     'postive', 'negative', 'never', 'umm', '!']
            print(f'Total of {len(valid)} features.')
            return dict([(k, i) for i, k in enumerate(valid)])
        elif method == 'threshold':
            # add words where probs greater than mean prob and less than
            # a fraction of the maximum
            minimum, maximum, mean, std = np.min(words[:, 1]), np.max(
                words[:, 1]), np.mean(words[:, 1]), np.std(words[:, 1])
            filtered = words[((words[:, 1] > mean*0.6) &
                              (words[:, 1] < maximum*0.01))]
            return dict([(k, i) for i, k in enumerate(filtered[:, 0])])
        elif method == 'length':
            # add words where their character length are greater than `threshold`
            threshold = 3
            lenghts = np.asarray([len(w) for w in words[:, 0]])
            filtered = words[(lenghts > threshold)]
            return dict([(k, i) for i, k in enumerate(filtered[:, 0])])
        elif method == 'ratio':
            # add words where the ratio of counts for
            # class i,j are greater than `threshold`
            threshold = 1.8
            filtered = []
            # top = []
            for i in self.class_dict.values():
                class_ratio = []
                for w_i, v_i in vs[classes[i]][0].items():
                    for j in self.class_dict.values():
                        if i == j:
                            continue
                        v_j = vs[classes[j]][0][w_i]  # returns 0 if KeyError
                        tmp = v_i if v_j == 0 else v_i/v_j
                        if tmp > threshold:
                            class_ratio.append((w_i, tmp))
                filtered.append(np.asarray(class_ratio, dtype=object)[:, 0])
                # top.append(class_ratio)
            filtered = np.hstack(filtered)
            #
            # top = np.vstack(top)
            # top = top[(-top[:, 1].astype(np.float32)).argsort()
            #           [:1000]]  # where [:n]
            # return dict([(k, i) for i, k in enumerate(top[:, 0])])
            #
            # remove words that have ratio > threshold in multiple classes
            # only occurs when there are more than 2 classes
            u, c = np.unique(filtered, return_counts=True)
            u = u[c == 1]
            return dict([(k, i) for i, k in enumerate(u)])
        elif method == 'remove':
            dic = self.select_features(vs, method='length_threshold')
            for w in [k for k in dic.keys() if any(s in k for s in ['\'', '--']) or
                      any(c.isdigit() for c in k)]:
                dic.pop(w, None)
            return dict([(k, i) for i, k in enumerate(dic)])
        elif method == 'default':
            # add default test words
            return {'fast': 0, 'couple': 1, 'shoot': 2, 'fly': 3}
        else:
            # combination of methods
            combination = []
            if 'ratio' in method:
                combination.append(self.select_features(vs, method='ratio'))
            if 'length' in method:
                combination.append(self.select_features(vs, method='length'))
            if 'threshold' in method:
                combination.append(
                    self.select_features(vs, method='threshold'))
            comb = set(combination[0].keys())
            for c in combination[1:]:
                comb &= set(c.keys())
            return dict([(k, i) for i, k in enumerate(comb)])


if __name__ == '__main__':
    start_t = time.time()
    nb = NaiveBayes()
    nb.train('movie_reviews/train')
    # nb.train('movie_reviews_small/train')
    print(f'Finished Training: {time.time() - start_t:.3f} s')
    results = nb.test('movie_reviews/dev')
    # results = nb.test('movie_reviews_small/test')
    nb.evaluate(results)
    print(f'Finished Training + Evaluation: {time.time() - start_t:.3f} s')
