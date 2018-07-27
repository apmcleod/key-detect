import numpy as np
import librosa
import os
import random
import math
import keys
from glob import glob
import madmom.audio.chroma


DATA_PREFIX = 'data/raw/'
GTZAN_PREFIX = DATA_PREFIX + 'genres/'
GTZAN_SUFFIX = '.au'
KEY_FILE_PREFIX = DATA_PREFIX + 'gtzan_key-master/gtzan_key/genres/'
KEY_FILE_SUFFIX = '.lerch.txt'

FS = 22050
LENGTH_GTZAN = 30
NUM_SAMPLES_GTZAN = FS * LENGTH_GTZAN

GENRES = ['country', 'pop', 'hiphop', 'reggae', 'classical', 'jazz', 'rock', 'blues', 'disco', 'metal']
GENRE_SIZES = [99, 94, 81, 97, 0, 79, 98, 98, 98, 93]
TOTAL_SIZE = np.sum(GENRE_SIZES)





def read_data(file):
    # Input: file name (relative to 'genres') directory, do read data from
    # Output: audio_data, y
    #        audio_data = numpy array containing each sample's value as a float vector
    #        y = normalized ground truth scoring vector for the given file *FROM get_vector_from_key method, above.
    y = keys.get_vector_from_key(int(open(KEY_FILE_PREFIX + file + KEY_FILE_SUFFIX, 'r').read()))
    y = np.reshape(y, (24, 1))
    audio_data, _ = librosa.core.load(GTZAN_PREFIX + file + GTZAN_SUFFIX, sr=FS, mono=True)
    audio_data = np.array(audio_data)
    audio_data = np.reshape(audio_data, (len(audio_data), 1))
    return audio_data, y



def cut_or_pad_to_length(vector, length):
    if vector.shape[0] != length:
        length_to_save = min(length, vector.shape[0])
        new_vector = np.zeros((length, 1))
        new_vector[:length_to_save, 0] = vector[:length_to_save, 0]
        return new_vector
    
    return vector



def load_all_data(directory):
    # Read all music files, and return them in arrays.
    # Output: audio_data, keys
    #        X = [num_samples, num_files] size matrix, containing the audio data, cut or padded with 0s to 30 seconds in length
    #        Y = [24, num_files] size matrix, containing the key vect
    file_list = [y for x in os.walk(directory) for y in glob(os.path.join(x[0], '*' + GTZAN_SUFFIX))]
    
    X = np.zeros((NUM_SAMPLES_GTZAN, 0))
    Y = np.zeros((24, 0))
    
    file_num = 0
    for file in file_list:
        if file_num % 10 == 0:
            print('Loading File ' + str(file_num) + '/' + str(len(file_list)))
        file_num += 1
        
        _, file_name = file.split('genres/')
        audio_data1, y1 = read_data(file_name[:-len(GTZAN_SUFFIX)])
        
        if np.sum(y1) == 0:
            print('WARNING: key unknown/modulation, skipping: file=' + file_name)
            continue
        
        audio_data1 = cut_or_pad_to_length(audio_data1, NUM_SAMPLES_GTZAN)
        
        X = np.append(X, audio_data1, axis=1)
        
        Y = np.append(Y, y1, axis=1)
        
    return X, Y



def write_np_data(X, Y, prefix):
    file_name = 'data/working/' + prefix + '.npz'
    
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))
        
    np.savez_compressed(file_name, X=X, Y=Y)



def load_np_data(prefix):
    loaded = np.load('data/working/' + prefix + '.npz')
    
    return loaded['X'], loaded['Y']



def process_data_into_np_files():
    if not os.path.exists('data/working/splits'):
        os.makedirs('data/working/splits')
        
    for genre in GENRES:
        print(genre)
        X, Y = load_all_data('data/raw/genres/' + genre)
        write_np_data(X, Y, genre)



def get_genre_and_song_idx_given_idx(idx, inclusive=True):
    if idx >= TOTAL_SIZE:
        if inclusive:
            return len(GENRE_SIZES) - 1, GENRE_SIZES[-1] - 1
        else:
            return len(GENRE_SIZES) - 1, GENRE_SIZES[-1]
        
    if idx <= 0:
        return 0, 0
    
    if not inclusive:
        genre_idx, song_idx = get_genre_and_song_idx_given_idx(idx - 1)
        return genre_idx, song_idx + 1
    
    for genre_idx in range(len(GENRES)):
        genre_size = GENRE_SIZES[genre_idx]
        genre_start_idx = int(np.sum(GENRE_SIZES[:genre_idx]))
        
        if genre_start_idx + genre_size > idx:
            return genre_idx, idx - genre_start_idx
    
    return len(GENRE_SIZES) - 1, GENRE_SIZES[-1]



def load_from_range(from_idx, to_idx):
    # Load the sample from a given index (inclusive) to a given index (exclusive)
    from_genre_idx, from_song_idx = get_genre_and_song_idx_given_idx(from_idx)
    to_genre_idx, to_song_idx = get_genre_and_song_idx_given_idx(to_idx, inclusive=False)
    
    #print('loading from ' + str((from_genre_idx, from_song_idx)) + ' to ' + str((to_genre_idx, to_song_idx)))
    
    X = np.zeros((NUM_SAMPLES_GTZAN, 0))
    Y = np.zeros((24, 0))
    
    for genre_idx in range(from_genre_idx, to_genre_idx + 1):
        genre_X, genre_Y = load_np_data(GENRES[genre_idx])
        
        to = genre_X.shape[1]
        if to_genre_idx == genre_idx:
            to = to_song_idx
        
        X = np.append(X, genre_X[:, from_song_idx:to], axis=1)
        Y = np.append(Y, genre_Y[:, from_song_idx:to], axis=1)
        
        from_song_idx = 0
    
    return X, Y



def load_song_by_idx(idx):
    return load_from_range(idx, idx + 1)




def create_random_splits(split_size=32):
    random.seed(1)
    song_indexes = list(range(TOTAL_SIZE))
    random.shuffle(song_indexes)

    for split_num in range(math.ceil(TOTAL_SIZE / split_size)):
        print('Making split ' + str(split_num) + '/' + str(TOTAL_SIZE // split_size))
        X = np.zeros((NUM_SAMPLES_GTZAN, 0))
        Y = np.zeros((24, 0))

        for song_idx_idx in range(split_size * split_num, min(split_size * (split_num + 1), TOTAL_SIZE)):
            if song_idx_idx % 10 == 0:
                print('Loading song idx ' + str(song_idx_idx))
            X_new, Y_new = load_song_by_idx(song_indexes[song_idx_idx])
            X = np.append(X, X_new, axis=1)
            Y = np.append(Y, Y_new, axis=1)

        write_np_data(X, Y, 'splits/split_' + str(split_num))
        
    print('Done')



def load_splits(indexes):
    # Load the splits from the given indexes
    X = np.zeros((NUM_SAMPLES_GTZAN, 0))
    Y = np.zeros((24, 0))
    
    for idx in indexes:
        X_new, Y_new = load_np_data('splits/split_' + str(idx))
        
        X = np.append(X, X_new, axis=1)
        Y = np.append(Y, Y_new, axis=1)
        
    return X, Y


def load_chroma_splits(indexes):
    # Load the chroma splits from the given indexes
    X = np.zeros((300, 12, 0))
    Y = np.zeros((24, 0))
    
    for idx in indexes:
        X_new, Y_new = load_np_data('chroma_splits/chroma_' + str(idx))
        
        X = np.append(X, X_new, axis=2)
        Y = np.append(Y, Y_new, axis=1)
        
    return X, Y


def generate_chroma_matrix_from_file(file):
    dcp = madmom.audio.chroma.DeepChromaProcessor()
    return dcp(file)


def generate_chroma_matrix_from_vector(X):
    librosa.output.write_wav('tmp.wav', np.reshape(X, -1), FS)
    
    chroma = generate_chroma_matrix_from_file('tmp.wav')
    
    os.remove('tmp.wav')
    return chroma


def generate_chroma_tensor_from_matrix(X):
    chroma = np.zeros((300, 12, X.shape[1]))
    
    for column in range(X.shape[1]):
        chroma[:, :, column] = generate_chroma_matrix_from_vector(X[:, column])
        
    return chroma


def generate_chroma_splits_from_splits():
    for split in range(27):
        print('Generating from split ' + str(split) + '/26')
        X, Y = load_splits([split])
        X = generate_chroma_tensor_from_matrix(X)
        
        write_np_data(X, Y, 'chroma_splits/chroma_' + str(split))
        