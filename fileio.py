#!/usr/bin/env python

import numpy as np
import pandas as pd
import librosa
import os
import shutil
import random
import math
import keys
import h5py
from glob import glob
import h5py


DATA_PREFIX = 'data'
RAW_PREFIX = '{}/raw'.format(DATA_PREFIX)
WORKING_PREFIX = '{}/working'.format(DATA_PREFIX)
CHUNK_PREFIX = '{}/chunks'.format(WORKING_PREFIX)
OUTPUT_PREFIX = '{}/output'.format(DATA_PREFIX)

CHUNK_SIZE = 100
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

GIANT_META = dict(
    NAME = 'giant',
    OUTFILE = 'labels_giant',
    DATA_PREFIX = RAW_PREFIX + '/giantsteps-key-dataset-master/audio',
    DATA_SUFFIX = '.LOFI.mp3',
    LABEL_PREFIX = RAW_PREFIX + '/giantsteps-key-dataset-master/annotations/key',
    LABEL_SUFFIX = '.LOFI.key',
    GENRE_PREFIX = RAW_PREFIX + '/giantsteps-key-dataset-master/annotations/genre',
    GENRE_SUFFIX = '.LOFI.genre'
)

GIANT_MTG_META = dict(
    NAME = 'giant_mtg',
    OUTFILE = 'labels_giant_mtg',
    DATA_PREFIX = RAW_PREFIX + '/giantsteps-mtg-key-dataset-master/audio',
    DATA_SUFFIX = '.LOFI.mp3',
    LABEL_PREFIX = RAW_PREFIX + '/giantsteps-mtg-key-dataset-master/annotations/key',
    LABEL_SUFFIX = '.LOFI.key',
    GENRE_PREFIX = RAW_PREFIX + '/giantsteps-mtg-key-dataset-master/annotations/genre',
    GENRE_SUFFIX = '.LOFI.genre'
)

MSD_META = dict(
    NAME = 'million',
    OUTFILE = 'labels_million',
    DATA_PREFIX = RAW_PREFIX + '/lmd_matched_mp3',
    DATA_SUFFIX = '.mp3',
    LABEL_PREFIX = RAW_PREFIX + '/lmd_matched_h5',
    LABEL_SUFFIX = '.h5',
    MIN_KEY_CONFIDENCE = 0.5,
    MIN_MODE_CONFIDENCE = 0.5
)


def read_audio_data(filepath, fs, duration=LENGTH):
    """
    Read audio data at a given rate from a given filepath
    
    Parameters
    ----------
    filepath : str
        the full path to the file to read
    fs : int
        the samplerate to import the file at
    duration: int
        max duration to read in seconds
    
    Returns
    -------
    audio_data : numpy array
        A numpy array containing the wav data
    
    """
    audio_data, _ = librosa.core.load(filepath, sr=fs, mono=True, duration=duration)
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

        
def validate_dir(path):
    """
    Checks whether a directory for writing to exists and queries 
    whether it should be deleted. Warns that files inside will be
    overwritten. This is used for the chunk directories.
    """
    if not os.path.exists(path):
        print('Creating directory for chunks: {}'.format(path))
        os.makedirs(path)
    else:
        print('Chunk directory exists {}.'.format(path))
        clear = input('Shall I clean it (y/n)? >')
        if clear == 'y':
            print('Cleaning chunk dir')
            shutil.rmtree(path)
            os.makedirs(path)
        else:
            print('WARNING: Overwriting old chunks (will not delete old chunks > max chunk nr)')
        print('Chunks will be saved in: {}'.format(path))


def load_all_data(chunk_size=CHUNK_SIZE, 
                  max_records=None,
                  datasets=['gtzan', 'giant', 'giant_mtg', 'msd'],
                  chunk_start_nr=0,
                  validate_chunk_dir=True,
                  chunk_prefix=CHUNK_PREFIX,
                  labels_prefix=WORKING_PREFIX):
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
    
    datasets : list of str
        list of datasets to import by name (order defined in function)
        
    chunk_start_nr : int
        a chunk number to start at
    
    validate_chunk_dir : bool
        whether to warn about the chunk directory
    
    chunk_prefix : str
        directory to save chunks to
        
    labels_prefix : str
        directory to save labels to
    
    Returns
    -------
    None
        Saves wav data files to '{}/chunk{:04d}.npz'.format(CHUNK_LOC, chunk_nr)
        (zipped numpy arrays) and '{}/labels_raw.pkl'.format(CHUNK_LOC) (a pickled pandas DataFrame)

    """
    if validate_chunk_dir:
        validate_dir(chunk_prefix)
    
    curr_chunk_nr = chunk_start_nr
    
    # Import GTZAN Data =======
    if 'gtzan' in datasets:
        labels_gtzan, curr_chunk_nr = load_gtzan_data(
                chunk_start_nr=curr_chunk_nr,
                chunk_size=chunk_size, 
                max_records=max_records,
                chunk_prefix=chunk_prefix,
                labels_prefix=labels_prefix)
    else:
        labels_gtzan = pd.read_pickle("{}/labels_gtzan.pkl".format(labels_prefix))
    
    # Import giantsteps data =======
    if 'giant' in datasets:
        labels_giant, curr_chunk_nr = load_giant_data(
                chunk_start_nr=curr_chunk_nr,
                chunk_size=chunk_size, 
                max_records=max_records,
                chunk_prefix=chunk_prefix,
                labels_prefix=labels_prefix)
    else:
        labels_giant = pd.read_pickle("{}/labels_giant.pkl".format(labels_prefix))
        
    # Import giantsteps-mtg data =======
    if 'giant_mtg' in datasets:
        labels_giant_mtg, curr_chunk_nr = load_giant_data(
                chunk_start_nr=curr_chunk_nr,
                chunk_size=chunk_size, 
                max_records=max_records,
                chunk_prefix=chunk_prefix,
                labels_prefix=labels_prefix,
                mtg=True)
    else:
        labels_giant_mtg = pd.read_pickle("{}/labels_giant_mtg.pkl".format(labels_prefix))
    
    
    # Import msd data =======
    if 'msd' in datasets:
        labels_msd, curr_chunk_nr = load_msd_data(
                chunk_start_nr=curr_chunk_nr,
                chunk_size=chunk_size, 
                max_records=max_records,
                chunk_prefix=chunk_prefix,
                labels_prefix=labels_prefix)
    else:
        labels_msd = pd.read_pickle("{}/labels_msd.pkl".format(labels_prefix))
    
    # Labels =======
    print('Writing labels')
    print('Labels saved in: {}'.format(labels_prefix))
    labels_raw = pd.concat([labels_gtzan, labels_giant, labels_giant_mtg, labels_msd], ignore_index=True)
    for col in ['key', 'chunk_idx', 'chunk_nr']:
        labels_raw[col] = labels_raw[col].astype('int')
    labels_raw.to_pickle('{}/labels_raw.pkl'.format(labels_prefix))
    
    print('FINISHED IMPORT!')
    
    
def load_gtzan_data(chunk_start_nr,
                    chunk_prefix,
                    labels_prefix,
                    chunk_size=CHUNK_SIZE, 
                    max_records=None):
    """
    See load_all_data for more info. Split these into seperate functions
    to allow for easier hacking of partial imports (and adding new data)
    """
    print('Processing GTZAN files {}'.format(20*'='))
    
    chunk_nr = chunk_start_nr
    labels = pd.DataFrame()
    meta = GTZAN_META
    
    pattern = os.path.join(meta['DATA_PREFIX'], '**', '*'+meta['DATA_SUFFIX'])
    # sorted important to preserve order
    file_list = sorted(glob(pattern, recursive=True))
    if max_records is not None:
        file_list = file_list[:max_records]
    for cc, files in enumerate(chunks(file_list, chunk_size)):
        chunk_nr = chunk_start_nr + cc
        delete_rows = []
        # use len(files) instead of chunk_size as last chunk will be smaller
        X_chunk = np.zeros((len(files), NUM_SAMPLES)) 
        chunk_name = 'chunk{:04d}'.format(chunk_nr)
        print('Processing chunk {} (size {})'.format(chunk_nr, len(files)))
        chunk_idx = 0
        for ii, filepath in enumerate(files):
            print('File {}/{}'.format(ii+1, len(files)), end="\r")
            file_stub = filepath.split('genres/')[1][:-3]
            label_path = os.path.join(meta['LABEL_PREFIX'], file_stub+meta['LABEL_SUFFIX'])
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
                time_shift = 1.0,
                chunk_nr = chunk_nr,
                chunk_idx = chunk_idx
            )
            chunk_idx += 1
            labels = labels.append(file_labels, ignore_index=True)
            audio_data = read_audio_data(filepath, FS)
            audio_data = cut_or_pad_to_length(audio_data, NUM_SAMPLES)
            X_chunk[ii, :] = audio_data
        X_chunk = np.delete(X_chunk, delete_rows, axis=0)
        file_name = '{}/{}.npz'.format(chunk_prefix, chunk_name)
        np.savez_compressed(file_name, X=X_chunk)
    labels['majmin'] = ['major' if key < 12 else 'minor' for key in labels['key']]
    labels['dataset'] = 'gtzan'
    labels.to_pickle('{}/labels_gtzan.pkl'.format(labels_prefix))
    return labels, chunk_nr+1
    

def load_giant_data(chunk_start_nr,
                    chunk_prefix,
                    labels_prefix,
                    chunk_size=CHUNK_SIZE, 
                    max_records=None,
                    mtg=False):
    """
    See load_all_data for more info. Split these into seperate functions
    to allow for easier hacking of partial imports (and adding new data)
    """
    print('Processing giant-steps files {}'.format(20*'='))
    
    chunk_nr = chunk_start_nr
    labels = pd.DataFrame()
    if mtg:
        meta = GIANT_MTG_META
    else:
        meta = GIANT_META
    
    pattern = os.path.join(meta['DATA_PREFIX'], '*'+meta['DATA_SUFFIX'])
    # sorted important to preserve order
    file_list = sorted(glob(pattern, recursive=True))
    if max_records is not None:
        file_list = file_list[:max_records]
    for cc, files in enumerate(chunks(file_list, chunk_size)):
        chunk_nr = chunk_start_nr + cc
        delete_rows = []
        # use len(files) instead of chunk_size as last chunk will be smaller
        X_chunk = np.zeros((len(files), NUM_SAMPLES)) 
        chunk_name = 'chunk{:04d}'.format(chunk_nr)
        print('Processing chunk {} (size {})'.format(chunk_nr, len(files)))
        chunk_idx = 0
        for ii, filepath in enumerate(files):
            print('File {}/{}'.format(ii+1, len(files)), end="\r")
            file_stub = filepath.rsplit('/', 1)[1].split('.', 1)[0]
            label_path = os.path.join(meta['LABEL_PREFIX'], file_stub+meta['LABEL_SUFFIX'])
            key_str = open(label_path).read().split('\t')[0]  # mtg changed the format >_<
            if key_str not in keys.KEY_DICT.keys():
                print('WARNING: invalid key [{}], skipping {}'.format(key_str, filepath))
                delete_rows += [ii]
                continue
            key = keys.KEY_DICT[key_str]
            genre_path = os.path.join(meta['GENRE_PREFIX'], file_stub+meta['GENRE_SUFFIX'])
            genre_str = open(genre_path).read().strip()
            file_labels = dict(
                filepath = filepath,
                genre = genre_str,
                key = key,
                key_str = keys.get_string_from_idx(key).replace('\t', ' '),
                raw = 1,
                key_shift = 0,
                time_shift = 1.0,
                chunk_nr = chunk_nr,
                chunk_idx = chunk_idx
            )
            chunk_idx += 1
            labels = labels.append(file_labels, ignore_index=True)
            audio_data = read_audio_data(filepath, FS)
            audio_data = cut_or_pad_to_length(audio_data, NUM_SAMPLES)
            X_chunk[ii, :] = audio_data
        X_chunk = np.delete(X_chunk, delete_rows, axis=0)
        file_name = '{}/{}.npz'.format(chunk_prefix, chunk_name)
        np.savez_compressed(file_name, X=X_chunk)
    labels['majmin'] = ['major' if key < 12 else 'minor' for key in labels['key']]
    labels['dataset'] = meta['NAME']
    labels.to_pickle('{}/{}.pkl'.format(labels_prefix, meta['OUTFILE']))
    return labels, chunk_nr+1


def load_msd_data(chunk_start_nr,
                  chunk_prefix,
                  labels_prefix,
                  chunk_size=CHUNK_SIZE, 
                  max_records=None):
    """
    See load_all_data for more info. Split these into seperate functions
    to allow for easier hacking of partial imports (and adding new data)
    """
    print('Processing MSD files {}'.format(20*'='))
    
    chunk_nr = chunk_start_nr
    labels = pd.DataFrame()
    meta = MSD_META
    
    pattern = os.path.join(meta['DATA_PREFIX'], '**', '*'+meta['DATA_SUFFIX'])
    # sorted important to preserve order
    file_list = sorted(glob(pattern, recursive=True))
    if max_records is not None:
        file_list = file_list[:max_records]
    for cc, files in enumerate(chunks(file_list, chunk_size)):
        chunk_nr = chunk_start_nr + cc
        delete_rows = []
        # use len(files) instead of chunk_size as last chunk will be smaller
        X_chunk = np.zeros((len(files), NUM_SAMPLES)) 
        chunk_name = 'chunk{:04d}'.format(chunk_nr)
        print('Processing chunk {} (size {})'.format(chunk_nr, len(files)))
        chunk_idx = 0
        for ii, filepath in enumerate(files):
            print('File {}/{}'.format(ii+1, len(files)), end="\r")
            file_stub = filepath.split('lmd_matched_mp3/')[1][:-len(meta['DATA_SUFFIX'])]
            label_path = os.path.join(meta['LABEL_PREFIX'], file_stub+meta['LABEL_SUFFIX'])
            
            key_map = h5py.File(label_path, 'r')['analysis']['songs']
            key = int(key_map['key'][0])
            key_confidence = key_map['key_confidence'][0]
            mode = int(key_map['mode'][0])
            mode_confidence = key_map['mode_confidence'][0]
            # Convert from C=0 (how h5 stores it) to A=0 (how we store it)
            key = keys.shift(key, 3)
            # Convert from maj=1, min=0 (how h5 stores it) to maj=[0-11], min=[12-23] (how we store it)
            key += (1 - mode) * 12
            
            if key_confidence < meta['MIN_KEY_CONFIDENCE'] or mode_confidence < meta['MIN_MODE_CONFIDENCE']:
                print('WARNING: key or mode uncertain ({}, {}), skipping {}.'.format(key_confidence, mode_confidence, filepath))
                delete_rows += [ii]   
                continue
            file_labels = dict(
                filepath = filepath,
                genre = file_stub.split('/')[0],
                key = key,
                key_str = keys.get_string_from_idx(key).replace('\t', ' '),
                raw = 1,
                key_shift = 0,
                time_shift = 1.0,
                chunk_nr = chunk_nr,
                chunk_idx = chunk_idx
            )
            chunk_idx += 1
            labels = labels.append(file_labels, ignore_index=True)
            audio_data = read_audio_data(filepath, FS)
            audio_data = cut_or_pad_to_length(audio_data, NUM_SAMPLES)
            X_chunk[ii, :] = audio_data
        X_chunk = np.delete(X_chunk, delete_rows, axis=0)
        file_name = '{}/{}.npz'.format(chunk_prefix, chunk_name)
        np.savez_compressed(file_name, X=X_chunk)
    labels['majmin'] = ['major' if key < 12 else 'minor' for key in labels['key']]
    labels['dataset'] = meta['NAME']
    labels.to_pickle('{}/{}.pkl'.format(labels_prefix, meta['OUTFILE']))
    return labels, chunk_nr+1


def make_labels(labels_prefix=WORKING_PREFIX):
    """
    This function is a placeholder for any label alteration that will
    need to happen due to data augmentation. For example, if data is 
    augmented by pitch shifting in a deterministic way, the labels_raw.pkl
    DataFrame can be copy pasted with small alterations and concatonated
    """
    df = pd.read_pickle("{}/labels_raw.pkl".format(labels_prefix))
    df.to_pickle("{}/labels.pkl".format(labels_prefix))
                     

def train_test_split(labels_prefix=WORKING_PREFIX, test_size=0.25, seed=42, shuffle=True):
    """
    Create a file called splits.npz containing train_idx and test_idx
    where test_size indicates the proportion of the set to use for
    testing.
    """
    np.random.seed(seed)
    df = pd.read_pickle("{}/labels_raw.pkl".format(labels_prefix))
    nr_obs = df.shape[0]
    size = int(nr_obs * test_size)
    test_idx = np.random.choice(nr_obs, size, replace=False)
    train_idx = np.array(list(set(range(nr_obs)).symmetric_difference(set(test_idx))))
    if shuffle:
        np.random.shuffle(test_idx)
        np.random.shuffle(train_idx)
    file_name = '{}/{}.npz'.format(labels_prefix, 'splits')
    np.savez_compressed(file_name, train_idx=train_idx, test_idx=test_idx)
    

def make_Y(labels_prefix=WORKING_PREFIX):
    """
    Creates the vectorised labels for training
    """
    df = pd.read_pickle("{}/labels_raw.pkl".format(labels_prefix))
    y = df['key']
    Y = np.array([keys.get_vector_from_key(kk) for kk in y])
    file_name = '{}/{}.npz'.format(labels_prefix, 'Y')
    np.savez_compressed(file_name, Y=Y)
    
    
def make_h5(labels_prefix=WORKING_PREFIX, h5_file='data.h5'):
    labels = pd.read_pickle("{}/labels.pkl".format(labels_prefix))

    with np.load("{}/splits.npz".format(labels_prefix)) as splits:
        train_idx = splits['train_idx']
        test_idx = splits['test_idx']

    X = np.load("{}/X_cqt.npz".format(labels_prefix))['X']
    X_train = X[train_idx, :]
    X_test = X[test_idx, :]

    Y = np.load("{}/Y.npz".format(labels_prefix))['Y']
    Y_train = Y[train_idx, :]
    Y_test = Y[test_idx, :]
    
    Y_test = np.argmax(Y_test, axis=1)
    Y_train = np.argmax(Y_train, axis=1)
    
    if os.path.exists('{}/{}'.format(labels_prefix, h5_file)):
        os.remove('{}/{}'.format(labels_prefix, h5_file))
    
    with h5py.File('{}/{}'.format(labels_prefix, h5_file), 'w') as file:
        file.create_dataset('X_train', data=X_train, dtype=float, chunks=(1,144,151), maxshape=(None, 144, 151))
        file.create_dataset('Y_train', data=Y_train, dtype='i8', chunks=(1024,), maxshape=(None,))
        file.create_dataset('X_test', data=X_test, dtype=float, chunks=(64,144,151), maxshape=(None, 144, 151))
        file.create_dataset('Y_test', data=Y_test, dtype='i8', chunks=(1024,), maxshape=(None,))

        
def augment_h5(labels_prefix=WORKING_PREFIX, h5_file='data.h5'):
    with h5py.File('{}/{}'.format(labels_prefix, h5_file), 'a') as file:
        with np.load('{}/data_aug.npz'.format(labels_prefix)) as aug:
            X_aug = aug['X']
            Y_aug = np.argmax(aug['Y'], axis=1)
            
            file['X_train'].resize(file['X_train'].shape[0] + X_aug.shape[0], axis=0)
            file['X_train'][-X_aug.shape[0]:] = X_aug
            
            file['Y_train'].resize(file['Y_train'].shape[0] + Y_aug.shape[0], axis=0)
            file['Y_train'][-Y_aug.shape[0]:] = Y_aug
        

if __name__ == "__main__":
    print("Performing initial data import")
    print("This expects data to have been downloaded as described in the README")
    load_all_data()
    make_labels()
    train_test_split()
    make_Y()
    
#     # Example of how to import a new datasource ('msd') without recreating old chunks
#     ## get chunk number to start on i.e. the last chunk number +1
#     chunk_start_nr = len([name for name in os.listdir(CHUNK_PREFIX)
#                           if os.path.isfile(name)])
#     ## You must call 'load_all_data', not 'load_msd_data' to correctly recreate the labels
#     load_all_data(chunk_size=CHUNK_SIZE, 
#                   max_records=None,
#                   datasets=['msd'],
#                   chunk_start_nr=chunk_start_nr,
#                   validate_chunk_dir=False,
#                   chunk_prefix=CHUNK_PREFIX,
#                   labels_prefix=WORKING_PREFIX)