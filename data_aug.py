#!/usr/bin/env python

import librosa
import librosa.core as lc
import fileio
import numpy as np
import pandas as pd
import subprocess
import math
import shutil
import features
import os
import keys
import time
import sys


SOX_PITCH_STRING = ' -400 -300 -200 -100 100 200 300 400 500 600 700 '
TMP_DATA = '{}/tmp'.format(fileio.DATA_PREFIX)

def augment_train(labels_prefix=fileio.WORKING_PREFIX, batch_size=50, batch_start=0, index_start=0, append=False):
    file_name = '{}/{}.npz'.format(labels_prefix, 'splits')
    train_idx = np.load(file_name)['train_idx'][index_start:]

    df = pd.read_pickle("{}/labels_raw.pkl".format(labels_prefix))
    filepaths = list(df['filepath'][train_idx])[index_start:]
    all_keys = list(df['key'][train_idx])[index_start:]

    num_batches = math.ceil(len(train_idx) / batch_size)
    
    for idx in range(batch_start, num_batches):
        print("Augmenting batch {}/{}".format(idx, num_batches - 1))
        
        X_aug = np.zeros((0, 144, fileio.LENGTH * 5 + 1))
        Y_aug = np.zeros((0, 24))
        
        if os.path.exists(TMP_DATA):
            shutil.rmtree(TMP_DATA)
        os.makedirs(TMP_DATA)

        bottom = idx * batch_size
        top = min((idx + 1) * batch_size, len(train_idx))

        files = filepaths[bottom : top]

        # Convert all mp3s to wavs
        print("Converting mp3s to wavs")
        for file_idx, file in enumerate(files):
            if file.endswith('.mp3'):
                audio_data = fileio.read_audio_data(file, fileio.FS)
                t = time.time()
                tmp_name = '{}/{}.wav'.format(TMP_DATA, t)
                librosa.output.write_wav(tmp_name, audio_data, fileio.FS)
                files[file_idx] = tmp_name

        train_keys = all_keys[bottom : top]

        file_list = ' ' + ' '.join(files) + ' '
        print("Calling sox")
        subprocess.call(['./augment.sh', TMP_DATA, file_list, SOX_PITCH_STRING])

        print("Reading outputs and calculating cqts")
        for file_idx, file in enumerate(files):
            file = os.path.basename(file)
            orig_key = train_keys[file_idx]

            for key_shift in [-4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7]:
                shifted_filename = '{}/{}.{}00.wav'.format(TMP_DATA, file, key_shift)
                shifted_key = keys.shift(orig_key, key_shift)

                shifted_audio_data = fileio.cut_or_pad_to_length(
                    fileio.read_audio_data(shifted_filename, fileio.FS), fileio.NUM_SAMPLES)

                cqt = features.get_cqt(shifted_audio_data)
                X_aug = np.concatenate((X_aug, cqt.reshape(1, cqt.shape[0], cqt.shape[1])), axis=0)
                Y_aug = np.concatenate((Y_aug, keys.get_vector_from_key(shifted_key).reshape(1, 24)), axis=0)
                
        np.savez_compressed('{}/data_aug_{}.npz'.format(labels_prefix, idx), X=np.log(X_aug + features.EPS), Y=Y_aug)

    if os.path.exists(TMP_DATA):
        shutil.rmtree(TMP_DATA)
    
    print('Joining all batch files')
    X_aug = np.zeros((0, 144, fileio.LENGTH * 5 + 1))
    Y_aug = np.zeros((0, 24))
    if append:
        with np.load("{}/data_aug.npz".format(DATA_DIR)) as aug:
            X_aug = aug['X']
            Y_aug = aug['Y']
        
    
    for idx in range(num_batches):
        file = '{}/data_aug_{}.npz'.format(labels_prefix, idx)
        if os.path.isfile(file):
            with np.load('{}/data_aug_{}.npz'.format(labels_prefix, idx)) as aug:
                X_aug = np.vstack((X_aug, aug['X']))
                Y_aug = np.vstack((Y_aug, aug['Y']))
        else:
            print('WARNING: {} not found. Skipping. (This may be normal with --append and --batch).'.format(file))
        
    print('Writing augmented file and cleaning augmented batches')
    
    np.savez_compressed('{}/data_aug.npz'.format(labels_prefix), X=X_aug, Y=Y_aug)
    
    for idx in range(num_batches):
        if os.path.isfile('{}/data_aug_{}.npz'.format(labels_prefix, idx)):
            os.remove('{}/data_aug_{}.npz'.format(labels_prefix, idx))





#################################### SHIFT PITCH AND TEMPO WITH LIBROSA
# To shift entire matrix X:
# X, X_shifted = pitch_and_tempo_shift_all(X)

PITCH_SHIFT_RANGE = list(range(0, 8)) + list(range(-4, 0))
TEMPO_SHIFT_RATES = np.array([0.9, 1.0, 1.1])

PITCH_SHIFT_RATES = (2.0 ** (-np.array(PITCH_SHIFT_RANGE, dtype=np.float) / 12))
RATES = np.dot(PITCH_SHIFT_RATES.reshape(-1, 1), TEMPO_SHIFT_RATES.reshape(1, -1))
RESAMPLE_RATES = fileio.FS / PITCH_SHIFT_RATES

NUM_SHIFTS = len(np.where(RATES != 1)[0])


#Outputs: X = original matrix, X_shifted_twice = augmented matrix, in the form:
# x_0,p0,T0.9
# x_0,p0,T1.1
# x_0,p1,T0.9
# x_0,p1,t1
# x_0,p1,t1.1
# x_0,p2,t0.9
# ...
def pitch_and_tempo_shift_batch(X):
    X_shifted = np.zeros((X.shape[0] * len(RATES), X.shape[1]))

    for idx in range(X.shape[0]):
        X_shifted[idx * len(RATES) : (idx + 1) * len(RATES), :] = pitch_and_tempo_shift_song(X[idx, :].reshape(-1))

    return X, X_shifted


def pitch_and_tempo_shift_song(X):
    X_shifted = np.zeros((len(RATES), len(X)))

    for idx in range(RATES.shape[0]):
        X_shifted[idx, :] = pitch_and_tempo_shift(stft, RATES[idx], RESAMPLE_RATES[idx])

    return X_shifted


def pitch_and_tempo_shift(stft, rate, resample_rate):
    stft_stretched = lc.phase_vocoder(stft, rate)
    X = lc.istft(stft_stretched)

    if resample_rate != fileio.FS:
        X = lc.resample(librosa.effects.time_stretch(X, rate), resample_rate, fileio.FS)

    return fileio.cut_or_pad_to_length(X, fileio.NUM_SAMPLES)


if __name__ == '__main__':
    print('Augmenting training data')
    batch_start = 0
    index_start = 0
    append = False
    for arg_idx in range(1, len(sys.argv)):
        arg = sys.argv[arg_idx]
        
        if arg == '--append' or arg == '-a':
            append = True
            print('Appending to existing data_aug.npz')
            
        elif arg.startswith('-b=') or arg.startswith('--batch='):
            batch_start = int(arg[arg.find('=') + 1 : ])
            print('Starting from batch {}'.format(batch_start))
        
        elif arg.startswith('-i=') or arg.startswith('--index='):
            index_start = int(arg[arg.find('=') + 1 : ])
            print('Starting from index {}'.format(batch_start))
        
        else:
            print('Data Augmentation usage error.')
            print('Usage: data_aug.py [-a (--append)] [-i=INT (--index=INT)] [-b=INT (--batch=INT)]')
            print()
            print('-a (--append): Append the created augmented data to the existing data_aug.npz file')
            print('-i=INT (--index=INT): Start augmenting songs from the given index (in the train_indexes list)') 
            print('-b=INT (--batch=INT): Start augmenting from the given batch number ' +
                  '(relative to the starting index from -i, if given)')
            print()
            print('WARNING: When using both -b and -i, the new batch files will be numbered from 0, beginning at index -i. ' +
                  'Only those batch numbers within the new batch range will be added to the final X_aug.')
            print()
            sys.exit(1)
    
    augment_train(batch_start=batch_start, index_start=index_start, append=append)
    print('Augmentations saved at {}/data_aug.npz'.format(fileio.WORKING_PREFIX))
    print('AUGMENTATION COMPLETE!')

