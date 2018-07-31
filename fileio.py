import numpy as np
import pandas as pd
import librosa
import os
import shutil
import random
import math
import keys
from glob import glob



DATA_PREFIX = 'data'
RAW_PREFIX = '{}/raw'.format(DATA_PREFIX)
WORKING_PREFIX = '{}/working'.format(DATA_PREFIX)
CHUNK_PREFIX = '{}/chunks'.format(WORKING_PREFIX)

CHUNK_SIZE = 250
FS = 22050
LENGTH = 30
NUM_SAMPLES = FS * LENGTH

GTZAN_META = dict(
    DATA_PREFIX = RAW_PREFIX + '/genres',
    DATA_SUFFIX = '.au',
    LABEL_PREFIX = RAW_PREFIX + '/gtzan_key-master/gtzan_key/genres',
    LABEL_SUFFIX = '.lerch.txt',
    GENRES = ['country', 'pop', 'hiphop', 'reggae', 'classical', 'jazz', 'rock', 'blues', 'disco', 'metal'],
    GENRE_SIZES = [99, 94, 81, 97, 0, 79, 98, 98, 98, 93],
    TOTAL_SIZE = np.sum([99, 94, 81, 97, 0, 79, 98, 98, 98, 93])
)


def read_audio_data(filepath, fs):
    """
    Read audio data at a given rate from a given filepath
    
    Parameters
    ----------
    filepath : str
        the full path to the file to read
    fs : int
        the samplerate to import the file at
    """
    audio_data, _ = librosa.core.load(filepath, sr=fs, mono=True)
    return audio_data


def cut_or_pad_to_length(vector, length):
    if vector.shape[0] != length:
        length_to_save = min(length, vector.shape[0])
        new_vector = np.zeros(length)
        new_vector[:length_to_save] = vector[:length_to_save]
        return new_vector
    else:
        return vector


def chunks(iterable, chunk_size):
    """
    Yield successive chunks from iterable
    
    Parameters
    ----------
    iterable : iterable
        list of other iterable to chunk up
    
    chunk_size : int
        The number of items in each yeild
    """
    for ii in range(0, len(iterable), chunk_size):
        yield iterable[ii : ii+chunk_size]


def load_all_data(chunk_size=CHUNK_SIZE, max_records=None):
    """
    Custom function which imports all data. Stores each music track as a wave vector of
    size D = FS*LENGTH within a matrix. Since data are large, data are split into chunks
    of shape CHUNK_SIZE x D. Imported wave data chunks are stored in CHUNK_LOC.
    
    This function also creates the label data for the corresponding audio. The index of
    each label corresponds to the (cumulative) index of the ordered chunks.
    
    Parameters
    ----------
    chunk_size : int
        The number of tracks to be stored in each chunk (default: CHUNK_SIZE)

    max_records : int
        For testing - max records to import
        
    Returns
    -------
    None
        Saves wav data files to '{}/chunk{:04d}.npz'.format(CHUNK_LOC, chunk_nr)
        (zipped numpy arrays) and '{}/labels_raw.pkl'.format(CHUNK_LOC) (a pickled pandas DataFrame)

    """
    # dataset, genre, key, key_str, majmin, raw, key_shift, time_shift
    labels_gtzan = pd.DataFrame()
    
    if not os.path.exists(CHUNK_PREFIX):
        print('Creating directory for chunks: {}'.format(CHUNK_PREFIX))
        os.makedirs(CHUNK_PREFIX)
    else:
        print('Chunk directory exists {}.'.format(CHUNK_PREFIX))
        clear = input('Shall I clean it (y/n)? >')
        if clear == 'y':
            print('Cleaning chunk dir')
            shutil.rmtree(CHUNK_PREFIX)
            os.makedirs(CHUNK_PREFIX)
        else:
            print('WARNING: Overwriting old chunks (will not delete old chunks > max chunk nr)')
        print('Chunks will be saved in: {}'.format(CHUNK_PREFIX))
    
    # Import GTZAN Data =======
    print('Processing GTZAN files {}'.format(20*'='))
    gtzan_pattern = os.path.join(GTZAN_META['DATA_PREFIX'], '**', '*'+GTZAN_META['DATA_SUFFIX'])
    # sorted important to preserve order
    file_list = sorted(glob(gtzan_pattern, recursive=True))
    if max_records is not None:
        file_list = file_list[:max_records]
    for chunk_nr, files in enumerate(chunks(file_list, chunk_size)):
        delete_rows = []
        # use len(files) instead of chunk_size as last chunk will be smaller
        X_chunk = np.zeros((len(files), NUM_SAMPLES)) 
        chunk_name = 'chunk{:04d}'.format(chunk_nr)
        print('Processing chunk {} (size {})'.format(chunk_nr, len(files)))
        for ii, filepath in enumerate(files):
            print('File {}/{}'.format(ii+1, len(files)), end="\r")
            file_stub = filepath.split('genres/')[1][:-3]
            label_path = os.path.join(GTZAN_META['LABEL_PREFIX'], file_stub+GTZAN_META['LABEL_SUFFIX'])
            key = int(open(label_path).read())
            if key == -1:
                print('WARNING: key unknown/modulation, skipping {}'.format(filepath))
                delete_rows += [ii]   
                continue
            file_labels = dict(
                filepath = filepath,
                genre = file_stub.split('/')[0],
                key = key,
                key_str = keys.get_string_from_idx(key).replace('\t', ' '),
                raw = 1,
                key_shift = 0,
                time_shift = 1.0
            )
            labels_gtzan = labels_gtzan.append(file_labels, ignore_index=True)
            audio_data = read_audio_data(filepath, FS)
            audio_data = cut_or_pad_to_length(audio_data, NUM_SAMPLES)
            X_chunk[ii, :] = audio_data
        X_chunk = np.delete(X_chunk, delete_rows, axis=0)
        file_name = '{}/{}.npz'.format(CHUNK_PREFIX, chunk_name)
        np.savez_compressed(file_name, X=X_chunk)
    labels_gtzan['majmin'] = ['major' if key < 12 else 'minor' for key in labels_gtzan['key']]
    labels_gtzan['dataset'] = 'GTZAN'
    labels_gtzan.to_pickle('{}/labels_gtzan.pkl'.format(WORKING_PREFIX))
      
    # Import giant-steps data =======
    print('Processing giant-steps files {}'.format(20*'='))
    labels_giant = pd.DataFrame()
    
    print('Writing labels')
    print('Labels saved in: {}'.format(WORKING_PREFIX))
    labels_raw = pd.concat([labels_gtzan, labels_giant], ignore_index=True)
    labels_raw['key'] = labels_raw['key'].astype('int')
    labels_raw.to_pickle('{}/labels_raw.pkl'.format(WORKING_PREFIX))
    
    print('FINISHED IMPORT!')