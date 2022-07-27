# Sample autograder for PA2
# =========================
#
# Place in same directory as logistic_regression.py to run
#
from logistic_regression import LogisticRegression
import numpy as np

# Track sumbission score
# score = 0
# max_score = 100

# def print_score():
#   print(f'Score: {score}/{max_score}\n')

# Function to replace featurize method in LogisticRegression Class


def featurize_test(document):
    feature_dict = {'fast': 0, 'couple': 1, 'shoot': 2, 'fly': 3}
    vector = np.zeros(len(feature_dict) + 1)
    for word in document:
        if word in feature_dict:
            vector[feature_dict[word]] += 1
    vector[-1] = 1
    return vector


print('1. Initialize NaiveBayes class from student solution... ', end='')

# import from student solution

# Initialize LogisticRegression object
lr = LogisticRegression(n_features=4)

# Override object attributes for use with test data
lr.featurize = featurize_test
print('PASSED')

print('2. Train classifier on movie_reviews_small dataset... ', end='')

lr.train('movie_reviews_small/train', batch_size=3, n_epochs=1, eta=0.1)
print('PASSED')

print('3. Testing theta... ', end='')

correct_theta = np.array(
    [-0.00770846,  0.04229154, -0.0812512,  0.0089582, -0.01500073])

if(np.allclose(lr.theta, correct_theta)):
    print('PASSED')
else:
    print('FAILED \nCorrect:')
    print(f'\tlr.theta = {correct_theta}')
    print('Output:')
    print(f'\tlr.theta = {lr.theta}')

print('4. Test results output from test method... ', end='')

correct_results = {'6.txt': {'correct': 0, 'predicted': 0}}

# Test against test set
results = lr.test('movie_reviews_small/test')

if(correct_results == results):
    print('PASSED')
else:
    print('FAILED')
    print('\nCorrect:')
    print(f'\tresults = {correct_results}')
    print('Output:')
    print(f'\tresults = {results}\n')

print('5. Test evaluate method output...\n(All metrics should show 0.5 or 50%!)\n')

results_test = {'1': {'correct': 0, 'predicted': 1},
                '2': {'correct': 0, 'predicted': 0},
                '3': {'correct': 1, 'predicted': 0},
                '4': {'correct': 1, 'predicted': 1}}
lr.evaluate(results_test)
