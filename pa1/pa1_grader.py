# Sample autograder for PA1
# =========================
#
# Place in same directory as naive_bayes.py to run
#
from naive_bayes import NaiveBayes
import numpy as np

# Track sumbission score
# score = 0
# max_score = 100

# def print_score():
#   print(f'Score: {score}/{max_score}\n')

print('1. Initialize NaiveBayes class from student solution... ', end='')


# Initialize object
nb = NaiveBayes()

# Override object attributes for use with test data
nb.class_dict = {'action': 0, 'comedy': 1}
nb.select_features = lambda *_: {'fast': 0, 'couple': 1, 'shoot': 2, 'fly': 3}
print('PASSED')

print('2. Train classifier on movie_reviews_small dataset... ', end='')

nb.train('movie_reviews_small/train')
print('PASSED')

print('3. Testing prior and likelihood... ', end='')

correct_prior = np.array([-0.51082562, -0.91629073])
correct_likelihood = np.array([[-1.79175947, -2.89037176, -1.28093385, -2.19722458],
                               [-2.07944154, -1.67397643, -2.77258872, -2.07944154]])

if(
    np.allclose(np.array(nb.prior), correct_prior) and
    np.allclose(np.array(nb.likelihood), correct_likelihood)
):
    print('PASSED')
else:
    print('FAILED \nCorrect:')
    print(f'\tnb.prior = {correct_prior}')
    print(f'\tnb.likelihood = \n{correct_likelihood}')
    print('Output:')
    print(f'\tnb.prior = {nb.prior}')
    print(f'\tnb.likelihood = \n{nb.likelihood}')

print('4. Test results output from test method... ', end='')

correct_results = dict({'6.txt': {'correct': 0, 'predicted': 0}})
results = nb.test('movie_reviews_small/test')

if(correct_results == results):
    print('PASSED')
else:
    print('FAILED')
    print('\nCorrect:')
    print(f'\tresults = {correct_results}')
    print('Output:')
    print(f'\tresults = {results}\n')

print('5. Test evaluate method output...\n')
nb.evaluate(correct_results)
print('\nPASSED')
