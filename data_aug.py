import librosa
import fileio
import numpy as np


# PITCH SHIFTING
# To shift entire matrices X, Y:
# X_shifted, Y_shifted = pitch_shift_all(X, Y, range(-4, 8))

def pitch_shift(X, Y, semitones):
    X = X.reshape(-1)
    Y = Y.reshape(Y.shape[0], 1)
    
    if semitones == 0:
        return X.reshape(X.shape[0], 1), Y
    
    X_shifted = librosa.effects.pitch_shift(X.reshape(-1), fileio.FS, n_steps=semitones).reshape(X.shape[0], 1)
    
    Y_shifted = np.zeros((Y.shape[0], 1))
    Y_shifted[:12, 0] = np.roll(Y[:12, 0], semitones)
    Y_shifted[12:24, 0] = np.roll(Y[12:24, 0], semitones)
    
    return X_shifted, Y_shifted


def pitch_shift_range(X, Y, semitone_range):
    X_shifted = np.zeros((X.shape[0], 0))
    Y_shifted = np.zeros((Y.shape[0], 0))
    
    for shift in semitone_range:
        X_shifted_once, Y_shifted_once = pitch_shift(X, Y, shift)
        X_shifted = np.append(X_shifted, X_shifted_once, axis=1)
        Y_shifted = np.append(Y_shifted, Y_shifted_once, axis=1)
        
    return X_shifted, Y_shifted


def pitch_shift_all(X, Y, semitone_range):
    X_shifted = np.zeros((X.shape[0], 0))
    Y_shifted = np.zeros((Y.shape[0], 0))
    
    for idx in range(X.shape[1]):
        X_shifted_once, Y_shifted_once = pitch_shift_range(X[:, idx], Y[:, idx], semitone_range)
        X_shifted = np.append(X_shifted, X_shifted_once, axis=1)
        Y_shifted = np.append(Y_shifted, Y_shifted_once, axis=1)
        
    return X_shifted, Y_shifted