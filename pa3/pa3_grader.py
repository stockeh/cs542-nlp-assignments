# Sample autograder for PA3
# =========================
#
# Place in same directory as pos_tagger.py to run
#
from pos_tagger import POSTagger
import numpy as np

# Track sumbission score
# score = 0
# max_score = 100

# def print_score():
#   print(f'Score: {score}/{max_score}\n')

print('1. Initialize POSTagger class from student solution...\n', end='')

# import from student solution

# Initialize POSTagger object
pos = POSTagger()

print('2. Check make_dicts on data_small dataset...\n', end='')

pos.make_dicts('data_small/train')

expected_tag_set = ['dt', 'nn', 'vb']
expected_word_set = ['Alice', 'Dorothy',
                     'admired', 'cheered', 'dwarf', 'every']
if (sorted(pos.tag_dict.keys()) == expected_tag_set and
        sorted(pos.word_dict.keys()) == expected_word_set):
    print('PASSED\n')
else:
    print('Correct tag set:', expected_tag_set)
    print('Output tag set:', sorted(pos.tag_dict.keys()))
    print('Correct word set:', expected_word_set)
    print('Output word set:', sorted(pos.word_dict.keys()))
    print('FAILED\n')

print('3. Check load_data on data_small training set...\n', end='')

sentence_ids, sentences, tag_lists, word_lists = pos.load_data(
    'data_small/train')

split_sentences = {}
for s in sentence_ids:
    split_sentences[s] = [word_tag.rsplit(
        '/', maxsplit=1) for word_tag in sentences[s].split()]

output_tag_lists = [[pos.tag_dict[word_tag[1]]
                     for word_tag in split_sentences[s]] for s in sentence_ids]
output_word_lists = [[pos.word_dict[word_tag[0]]
                      for word_tag in split_sentences[s]] for s in sentence_ids]

if len(output_tag_lists) > 0 and \
        all([[pos.tag_dict[word_tag[1]] for word_tag in split_sentences[s]] ==
            tag_lists[s] for s in sentence_ids]) and \
        len(output_word_lists) > 0 and \
        all([[pos.word_dict[word_tag[0]] for word_tag in split_sentences[s]] == word_lists[s] for s in sentence_ids]):
    print('PASSED\n')
else:
    if len(output_tag_lists) == 0 or len(output_word_lists) == 0:
        print('Output tag lists and output word lists cannot be empty!')
    else:
        print('Correct tag lists:', [tag_lists[s] for s in sentence_ids])
        print('Output tag lists:', output_tag_lists)
        print('Correct word lists:', [word_lists[s] for s in sentence_ids])
        print('Output word lists:', output_word_lists)
    print('FAILED\n')

print('4. Testing train on dummy data...\n', end='')

sentence_ids = ['ca010', 'ca030', 'ca040']
sentences = {'ca010': 'Alice/nn admired/vb Dorothy/nn',
             'ca030': 'every/dt dwarf/nn cheered/vb',
             'ca040': 'Dorothy/nn admired/vb every/dt dwarf/nn'}
tag_lists = {'ca010': [0, 1, 0], 'ca030': [2, 0, 1], 'ca040': [0, 1, 2, 0]}
word_lists = {'ca010': [0, 1, 2], 'ca030': [3, 4, 5], 'ca040': [0, 1, 3, 4]}

pos.tag_dict = {'nn': 0, 'vb': 1, 'dt': 2}
pos.word_dict = {'Alice': 0, 'admired': 1, 'Dorothy': 2, 'every': 3,
                 'dwarf': 4, 'cheered': 5}
pos.initial = np.array([-0.3, -0.7, 0.3])
pos.transition = np.array([[-0.7, 0.3, -0.3],
                           [-0.3, -0.7, 0.3],
                           [0.3, -0.3, -0.7]])
pos.emission = np.array([[-0.3, -0.7, 0.3],
                         [0.3, -0.3, -0.7],
                         [-0.3, 0.3, -0.7],
                         [-0.7, -0.3, 0.3],
                         [0.3, -0.7, -0.3],
                         [-0.7, 0.3, -0.3]])

pos.train('data_small/train', dummy_data=(sentence_ids,
          sentences, tag_lists, word_lists))

expected_initial = np.array([0.7, -1.7, 0.3])
expected_transition = np.array([[-0.7, 0.3, -0.3],
                                [-0.3, -0.7, 0.3],
                                [0.3, -0.3, -0.7]])
expected_emission = np.array([[0.7, -0.7, -0.7],
                              [-0.7, 0.7, -0.7],
                              [0.7, -0.7, -0.7],
                              [-0.7, -1.3, 1.3],
                              [0.3, -0.7, -0.3],
                              [-0.7, 0.3, -0.3]])

if (np.allclose(pos.initial, expected_initial) and
    np.allclose(pos.transition, expected_transition) and
        np.allclose(pos.emission, expected_emission)):
    print('PASSED\n')
else:
    print('Correct initial weights:', expected_initial)
    print('Output initial weights:', pos.initial)
    print('Correct transition weights:', expected_transition)
    print('Output transition weights:', pos.transition)
    print('Correct emission weights:', expected_emission)
    print('Output emission weights:', pos.emission)
    print('FAILED\n')

print('5. Testing viterbi on dummy data...\n', end='')

expected_best_path = [0, 1, 2, 0]
best_path = pos.viterbi(word_lists['ca040'])

if (best_path == expected_best_path):
    print('PASSED\n')
else:
    print('Correct best path:', expected_best_path)
    print('Output best path:', best_path)
    print('FAILED\n')

print('6. Testing test on dummy data...\n', end='')

sentence_ids = ['ca020', 'ca050']
sentences = {'ca020': 'Alice/nn cheered/vb',
             'ca050': 'Goldilocks/nn cheered/vb'}
tag_lists = {'ca020': [0, 1], 'ca050': [0, 1]}
word_lists = {'ca020': [0, 5], 'ca050': [-1, 5]}

sentences, results = pos.test(
    'data_small/test', dummy_data=(sentence_ids, sentences, tag_lists, word_lists))

expected_test = {'correct': [0, 1], 'predicted': [0, 1]}

# cs050 contains unknown word, don't check against it
if expected_test == results['ca020']:
    print('PASSED\n')
else:
    print('Correct:', expected_test)
    print('Output:', results['ca020'])
    print('FAILED\n')

print('7. Test evaluate method output on dummy data...\n', end='')

results_test = {'1': {'correct': [0, 1], 'predicted': [0, 1]},
                '2': {'correct': [0, 2, 0], 'predicted': [0, 2, 1]},
                '3': {'correct': [1, 2, 3], 'predicted': [3, 2, 1]},
                '4': {'correct': [0, 0], 'predicted': [1, 1]}}
accuracy = pos.evaluate(sentences, results_test, dummy_data=True)

if accuracy == 0.5:
    print('PASSED\n')
else:
    print('Correct:', 0.5)
    print('Output:', accuracy)
    print('FAILED\n')
