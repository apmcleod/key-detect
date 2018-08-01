import librosa
import librosa.core as lc
import fileio
import numpy as np
import pandas as pd
import subprocess
import math
import shutil
import features


SOX_PITCH_STRING = '"-400 -300 -200 -100 100 200 300 400 500 600 700"'
TMP_DATA = '{}/tmp'.format(fileio.DATA_PREFIX)

def augment_train(labels_prefix=fileio.WORKING_PREFIX, batch_size=50):
    file_name = '{}/{}.npz'.format(labels_prefix, 'splits')
    train_idx = np.load(file_name)['train_idx']
    
    df = pd.read_pickle("{}/labels_raw.pkl".format(labels_prefix))
    filepaths = list(df['filepath'][train_idx])
    all_keys = list(df['key'][train_idx])
        
    num_batches = math.ceil(len(train_idx) / batch_size)
    
    X_aug = np.zeros((0, 144, fileio.LENGTH * 5 + 1))
    Y_aug = np.zeros((0, 24))
    
    for idx in range(num_batches):
        print("Augmenting batch {}/{}".format(idx, num_batches))
        shutil.rmtree(TMP_DATA)
        os.makedirs(TMP_DATA)
        
        bottom = idx * batch_size
        top = min((idx + 1) * batch_size, len(train_idx))
        
        files = filepaths[bottom : top]
        keys = all_keys[bottom : top]
        
        file_list = '"' + ' '.join(files) + '"'
        subprocess.call('./augment.sh {} {} {}'.format(TMP_DATA, file_list, SOX_PITCH_STRING))
        
        for file_idx, file in enumerate(files):
            orig_key = keys[file_idx]
            
            for key_shift in [-4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7]:
                shifted_filename = '{}.{}00'
                shifted_key = keys.shift(orig_key, key_shift)
                shifted_audio_data = fileio.cut_or_pad_to_length(
                    fileio.NUM_SAMPLES, fileio.read_audio_data(shifted_filename, fileio.FS))
                
                cqt = features.get_cqt(shifted_audio_data)
                X_aug = np.concatenate((X_aug, cqt.reshape(1, cqt.shape[0], cqt.shape[1])), axis=0)
                Y_aug = np.concatenate((Y_aug, keys.get_vector_from_key(shifted_key)), axis=0)
                
    shutil.rmtree(TMP_DATA)
    
    X_aug = np.log(X_aug + features.EPS)
    
    np.savez_compressed('{}/data_aug.npz'.format(labels_prefix), X=X_aug, Y=Y_aug





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
