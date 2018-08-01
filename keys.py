import numpy as np

STRING_MAP = ['A\tmajor', 'A#\tmajor', 'B\tmajor', 'C\tmajor', 'C#\tmajor',
             'D\tmajor', 'D#\tmajor', 'E\tmajor', 'F\tmajor', 'F#\tmajor',
             'G\tmajor', 'G#\tmajor', 'A\tminor', 'A#\tminor', 'B\tminor',
             'C\tminor', 'C#\tminor', 'D\tminor', 'D#\tminor', 'E\tminor',
             'F\tminor', 'F#\tminor', 'G\tminor', 'G#\tminor']

# for maping giant dataset labels to numbers
note_letters = list('ABCDEFGabcdefg')
note_numbers = [0, 2, 3, 5, 7, 8, 10] * 2
for ii, letter in enumerate(list('ABCDEFGabcdefg')):
    note_letters.append('{}#'.format(letter))
    note_numbers.append((note_numbers[ii]+1) % 12)
    note_letters.append('{}b'.format(letter))
    note_numbers.append((note_numbers[ii]-1) % 12)
key_strings = ['{} major'.format(ll) for ll in note_letters] + ['{} minor'.format(ll) for ll in note_letters]
key_numbers = note_numbers + [nn + 12 for nn in note_numbers]
KEY_DICT = dict(zip(key_strings, key_numbers))

KEY_SUM = 1 + 0.5 + 0.5 + 0.3 + 0.2

def get_vector_from_key(key):
    vector = np.zeros(24)
    if key == -1: # Unknown
        return vector
    if key < 12: #major
        vector[key] = 1
        vector[(key + 7) % 12] = 0.5
        vector[(key + 5) % 12] = 0.5
        vector[(key + 9) % 12 + 12] = 0.3
        vector[key + 12] = 0.2
    else: # minor
        vector[key] = 1
        vector[(key + 7) % 12 + 12] = 0.5
        vector[(key + 5) % 12 + 12] = 0.5
        vector[(key + 3) % 12] = 0.3
        vector[key - 12] = 0.2

    vector /= KEY_SUM

    return vector

def get_string_from_idx(idx):
    return STRING_MAP[idx]

def get_string_from_vector(vector):
    return STRING_MAP[np.argmax(vector)]


def generate_one_hot_guess(vector):
    return np.argmax(vector)


def generate_one_hot_matrix(matrix):
    return np.argmax(matrix, axis=0)

def shift(key, shift):
    maj = key // 12
    num = key % 12
    
    num = (num + shift + 12) % 12
    
    return num + maj * 12
