# Sample autograder for PA5
# =========================
#
# Place in same directory as parser .py files to run
#
from run import load_and_preprocess_data, train
from parser_transitions import test_minibatch_parse
from parser_transitions import test_parse
from parser_transitions import test_parse_step
from parser_model import ParserModel
import numpy as np
import time
from datetime import datetime
import os

# Track sumbission score
score = 0
max_score = 65


def print_score():
    print(f'Score: {score}/{max_score}\n')


print('1. Importing ParserModel from student solution...\n')

# import from student solution
model = ParserModel(np.zeros((100, 30), dtype=np.float32))
print("PASSED")
score += 5

print('\n2. Testing parser_transitions.test_parse_step...\n')


if (test_parse_step()):
    print("PASSED")
    score += 10
else:
    print("FAILED")

print('\n3. Testing parser_transitions.test_parse...\n')


if (test_parse()):
    print("PASSED")
    score += 10
else:
    print("FAILED")

print('\n4. Testing parser_transitions.test_minibatch_parse...\n')


if (test_minibatch_parse()):
    print("PASSED")
    score += 10
else:
    print("FAILED")

print('\n5. Testing embeddings lookup...\n')

if (model.check_embedding()):
    print("PASSED")
    score += 5
else:
    print("FAILED")

print('\n6. Testing forward...\n')

if (model.check_forward()):
    print("PASSED")
    score += 5
else:
    print("FAILED")

print('\n7. Full debug run...\n')


print(80 * "=")
print("INITIALIZING")
print(80 * "=")
parser, embeddings, train_data, dev_data, test_data = load_and_preprocess_data(
    True)

start = time.time()
model = ParserModel(embeddings)
parser.model = model
print("took {:.2f} seconds\n".format(time.time() - start))

print(80 * "=")
print("TRAINING")
print(80 * "=")
output_dir = "results/{:%Y%m%d_%H%M%S}/".format(datetime.now())
output_path = output_dir + "model.weights"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

uas = train(parser, train_data, dev_data, output_path,
            batch_size=1024, n_epochs=10, lr=0.0005)

pp = np.round((.7-uas)*20 if uas < .7 else 0)
print("Performance penalty: %.0f" % (pp,))

score += 20-pp
print_score()
