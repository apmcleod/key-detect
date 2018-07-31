import librosa
import librosa.core as lc
import fileio
import numpy as np


# SHIFT PITCH AND TEMPO
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
