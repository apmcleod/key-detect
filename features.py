#!/usr/bin/env python
import fileio
import numpy as np
import librosa
from glob import glob

EPS = np.finfo(float).eps


def cqt():
    print('Creating cqt features from chunked audio')
    
    X_cqt = np.zeros((0, 144, fileio.LENGTH * 5 + 1))
    
    chunk_files = sorted(glob('{}/*.npz'.format(fileio.CHUNK_PREFIX), recursive=True))
    nr_chunks = len(chunk_files)
    
    for ii, chunk in enumerate(chunk_files):
        print('Processing chunk {}/{}'.format(ii+1, nr_chunks), end='\r')
        X = np.load(chunk)['X']

        # Data aug would be here, but too slow:
        # X, X_shifted = data_aug.pitch_and_tempo_shift_batch(X)
        # X = np.append(X, X_shifted, axis=0)

        for song in X:
            cqt = librosa.core.hybrid_cqt(song, sr=fileio.FS, bins_per_octave=24, n_bins=144, hop_length=int(fileio.FS / 5))
            X_cqt = np.concatenate((X_cqt, cqt.reshape(1, cqt.shape[0], cqt.shape[1])), axis=0)

    np.savez_compressed('{}/X_cqt.npz'.format(fileio.WORKING_PREFIX), X=np.log(X_cqt + EPS))

if __name__ == '__main__':
    cqt()
