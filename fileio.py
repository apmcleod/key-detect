#!/usr/bin/env python

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
                  datasets=['gtzan', 'giant', 'msd'],
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
    
    # Import giant-steps data =======
    if 'giant' in datasets:
        labels_giant, curr_chunk_nr = load_gtzan_data(
                chunk_start_nr=curr_chunk_nr,
                chunk_size=chunk_size, 
                max_records=max_records,
                chunk_prefix=chunk_prefix,
                labels_prefix=labels_prefix)
    else:
        labels_giant = pd.read_pickle("{}/labels_giant.pkl".format(labels_prefix))
    
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
    labels_raw = pd.concat([labels_gtzan, labels_giant, labels_msd], ignore_index=True)
    labels_raw['key'] = labels_raw['key'].astype('int')
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
    chunk_nr = chunk_start_nr
    labels_gtzan = pd.DataFrame()
    
    print('Processing GTZAN files {}'.format(20*'='))
    gtzan_pattern = os.path.join(GTZAN_META['DATA_PREFIX'], '**', '*'+GTZAN_META['DATA_SUFFIX'])
    # sorted important to preserve order
    file_list = sorted(glob(gtzan_pattern, recursive=True))
    if max_records is not None:
        file_list = file_list[:max_records]
    for cc, files in enumerate(chunks(file_list, chunk_size)):
        chunk_nr = chunk_start_nr + cc
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
                time_shift = 1.0,
                chunk_nr = chunk_nr
            )
            labels_gtzan = labels_gtzan.append(file_labels, ignore_index=True)
            audio_data = read_audio_data(filepath, FS)
            audio_data = cut_or_pad_to_length(audio_data, NUM_SAMPLES)
            X_chunk[ii, :] = audio_data
        X_chunk = np.delete(X_chunk, delete_rows, axis=0)
        file_name = '{}/{}.npz'.format(chunk_prefix, chunk_name)
        np.savez_compressed(file_name, X=X_chunk)
    labels_gtzan['majmin'] = ['major' if key < 12 else 'minor' for key in labels_gtzan['key']]
    labels_gtzan['dataset'] = 'GTZAN'
    labels_gtzan.to_pickle('{}/labels_gtzan.pkl'.format(labels_prefix))
    return labels_gtzan, chunk_nr+1
    

def load_giant_data(chunk_start_nr,
                    chunk_prefix,
                    labels_prefix,
                    chunk_size=CHUNK_SIZE, 
                    max_records=None):
    """
    See load_all_data for more info. Split these into seperate functions
    to allow for easier hacking of partial imports (and adding new data)
    """
    print('Processing giant-steps files {}'.format(20*'='))
    chunk_nr = chunk_start_nr
    labels_giant = pd.DataFrame()
#     return labels_gtzan, chunk_nr+1
    return labels_giant, chunk_nr


def load_msd_data(chunk_start_nr,
                  chunk_prefix,
                  labels_prefix,
                  chunk_size=CHUNK_SIZE, 
                  max_records=None):
    """
    See load_all_data for more info. Split these into seperate functions
    to allow for easier hacking of partial imports (and adding new data)
    """
    print('Processing giant-steps files {}'.format(20*'='))
    chunk_nr = chunk_start_nr
    labels_msd = pd.DataFrame()
#     return labels_gtzan, chunk_nr+1
    return labels_msd, chunk_nr


if __name__ == "__main__":
    print("Performing initial data import")
    print("This expects data to have been downloaded as described in the README")
    load_all_data()
    
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