import os


class Stemmer():
    '''Modified Porter Word Stemming Algorithm.
    Code adopted from Jason Stock (https://github.com/stockeh/search-engine/blob/master/src/stemming.cpp)
    and converted from C++ to Python.
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


if __name__ == '__main__':
    stemmer = Stemmer(exception_path='Exceptions.txt')
    ws = {'skies'}
    for w in ws:
        print(w, '->', stemmer.stem(w))
    # stemmer = Stemmer('Exceptions.txt', inputdir='movie_reviews/train/pos')
    # output = stemmer.iterate_documents()
    # print(list(iter(output.values()))[:1])
