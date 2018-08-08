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
import multiprocessing as mp




def augment_from_pkl(pkl_path='{}/augment.pkl'.format(fileio.WORKING_PREFIX), num_procs=1):
    to_augment = pd.read_pickle(pkl_path)
    
    # Shuffle here, because minor songs get augmented more than major songs (i.e. their to_keys array is longer).
    # This randomizes so that each parallel process is doing the same amount of work
    to_augment = to_augment.sample(frac=1, random_state=1)
    
    total_size = len(to_augment)
    batch_size = math.ceil(total_size / num_procs)
    
    processes = [mp.Process(
        target=augment_from_dataframe_parallel,
        args=(
            proc_id,
            to_augment[proc_id * batch_size : min((proc_id + 1) * batch_size, total_size)]))
        for proc_id in range(num_procs)]
    
    if os.path.exists(TMP_DATA):
        shutil.rmtree(TMP_DATA)
    os.makedirs(TMP_DATA)
    
    for p in processes:
        p.start()
        
    for p in processes:
        p.join()
    
    X_aug = np.zeros((0, 144, fileio.LENGTH * 5 + 1))
    Y_aug = np.zeros((0, 24))
    
    for proc_id in range(num_procs):
        with np.load('{}/data_aug_{}.npz'.format(fileio.WORKING_PREFIX, proc_id)) as aug:
            X_aug = np.vstack((X_aug, aug['X']))
            Y_aug = np.vstack((Y_aug, aug['Y']))
            
    np.savez_compressed('{}/data_aug.npz'.format(fileio.WORKING_PREFIX), X=X_aug, Y=Y_aug)
    
    for proc_id in range(num_procs):
        if os.path.isfile('{}/data_aug_{}.npz'.format(fileio.WORKING_PREFIX, proc_id)):
            os.remove('{}/data_aug_{}.npz'.format(fileio.WORKING_PREFIX, proc_id))
        
    if os.path.exists(TMP_DATA):
        shutil.rmtree(TMP_DATA)
        
    print('All done!')
    
    

def augment_from_dataframe_parallel(proc_id, to_augment):
    X_aug = np.zeros((0, 144, fileio.LENGTH * 5 + 1))
    Y_aug = np.zeros((0, 24))
    
    for idx, (filename, row) in enumerate(to_augment.iterrows()):
        if idx % 100 == 0:
            print('Process {} augmenting file {}/{}'.format(proc_id, idx, len(to_augment)))
            
        shifts = row[0]
        orig_key = row[1]
        
        sox_shifts = ' ' + ' '.join(str(shift * 100) for shift in shifts) + ' '
        
        # Convert from mp3 if needed
        if filename.endswith('.mp3'):
            audio_data = fileio.read_audio_data(filename, fileio.FS)
            t = time.time()
            tmp_name = '{}/{}_{}.wav'.format(TMP_DATA, proc_id, t)
            librosa.output.write_wav(tmp_name, audio_data, fileio.FS)
            filename = tmp_name
        
        # Shift with sox
        subprocess.call(['./augment.sh', TMP_DATA, filename, sox_shifts])
        os.remove(filename)

        basename = os.path.basename(filename)
        
        # Read shifted files
        for key_shift in shifts:
            shifted_filename = '{}/{}.{}00.wav'.format(TMP_DATA, basename, key_shift)
            shifted_key = keys.shift(orig_key, key_shift)

            shifted_audio_data = fileio.cut_or_pad_to_length(
                fileio.read_audio_data(shifted_filename, fileio.FS), fileio.NUM_SAMPLES)
            os.remove(shifted_filename)

            cqt = features.get_cqt(shifted_audio_data)
            X_aug = np.concatenate((X_aug, cqt.reshape(1, cqt.shape[0], cqt.shape[1])), axis=0)
            Y_aug = np.concatenate((Y_aug, keys.get_vector_from_key(shifted_key).reshape(1, 24)), axis=0)
            
    np.savez_compressed('{}/data_aug_{}.npz'.format(fileio.WORKING_PREFIX, proc_id), X=np.log(X_aug + features.EPS), Y=Y_aug)
    print('Process {} done!'.format(proc_id))

    
def create_augment_pkl(labels_prefix=fileio.WORKING_PREFIX, pkl_name='augment.pkl'):
    train_idx = np.load('{}/{}.npz'.format(labels_prefix, 'splits'))['train_idx']
    df = pd.read_pickle("{}/labels_raw.pkl".format(labels_prefix))
    
    pkl_path = '{}/{}'.format(labels_prefix, pkl_name)
    
    augment_counts = get_augment_counts(df, train_idx)
    
    shift_dictionary = {}
    
    for key_from in range(24):
        key_df = df[df['key']==key_from]
        for key_to in np.where(augment_counts[key_from] != 0)[0]:
            shift_amount = (key_to - key_from + 12) % 12
            if shift_amount > 7:
                shift_amount -= 12
                
            songs_to_shift = key_df.sample(n=augment_counts[key_from, key_to], random_state=key_to * key_from)
            
            for filename in songs_to_shift['filepath']:
                if filename in shift_dictionary:
                    shift_dictionary[filename][0] = np.append(shift_dictionary[filename][0], shift_amount)
                else:
                    shift_dictionary[filename] = [np.array([shift_amount]), key_from]
                    
    pd.DataFrame.from_dict(shift_dictionary, orient='index').to_pickle(pkl_path)
    

    
def get_augment_counts(df, train_idx):
    counts = df.loc[train_idx, 'key'].value_counts().sort_index()
    max_count = df.loc[train_idx, 'key'].value_counts().max()
    
    augment_counts = np.zeros((24, 24), dtype=np.int)
    
    for key in range(24):
        augment_key(key, augment_counts, counts, max_count)
        
    return augment_counts



AUGMENT_ORDER = [1, -1, 2, -2, 3, -3, 4, -4, 5, 6, 7]

def augment_key(key, augment_counts, counts, max_count):
    key_count = counts[key]
    
    for augment in AUGMENT_ORDER:
        if key_count == max_count:
            break
        
        diff = max_count - key_count
        
        key_to_augment_from = keys.shift(key, -augment)
        count_to_augment = int(min(diff, counts[key_to_augment_from]))
        
        augment_counts[key_to_augment_from, key] = count_to_augment
        key_count += count_to_augment
        


        
SOX_PITCH_STRING = ' -400 -300 -200 -100 100 200 300 400 500 600 700 '
TMP_DATA = '{}/tmp'.format(fileio.DATA_PREFIX)

def augment_train_full(labels_prefix=fileio.WORKING_PREFIX, batch_size=50, batch_start=0, index_start=0, append=False):
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




if __name__ == '__main__':
    print('Augmenting training data')
    batch_start = 0
    index_start = 0
    num_procs = 1
    append = False
    full = False
    for arg_idx in range(1, len(sys.argv)):
        arg = sys.argv[arg_idx]
        
        if arg == '--append' or arg == '-a':
            append = True
            print('Appending to existing data_aug.npz')
            
        elif arg == '--full' or arg == '-f-':
            full = True
            print('Augmenting all songs to full range')
            
        elif arg.startswith('-b=') or arg.startswith('--batch='):
            batch_start = int(arg[arg.find('=') + 1 : ])
            print('Starting from batch {}'.format(batch_start))
        
        elif arg.startswith('-i=') or arg.startswith('--index='):
            index_start = int(arg[arg.find('=') + 1 : ])
            print('Starting from index {}'.format(batch_start))
            
        elif arg.startswith('-p=') or arg.startswith('--procs='):
            num_procs = int(arg[arg.find('=') + 1 : ])
            print('Running with {} processes'.format(num_procs))
        
        else:
            print('Data Augmentation usage error.')
            print('Usage: data_aug.py [-p=INT (--procs==INT)] [-f (--full) [-a (--append)] [-i=INT (--index=INT)] [-b=INT (--batch=INT)]]')
            print()
            print('-p=INT (--procs==INT): Run sample-based augmentation in parallel with INT processes. (Not used with -f).')
            print()
            print('-f (--full): Augment all songs to the full [-4, +7] range.')
            print()
            print('             With -f, the following options are also available:')
            print('-a (--append): Append the created augmented data to the existing data_aug.npz file')
            print('-i=INT (--index=INT): Start augmenting songs from the given index (in the train_indexes list)') 
            print('-b=INT (--batch=INT): Start augmenting from the given batch number ' +
                  '(relative to the starting index from -i, if given)')
            print()
            print('WARNING: When using both -b and -i, the new batch files will be numbered from 0, beginning at index -i. ' +
                  'Only those batch numbers within the new batch range will be added to the final X_aug.')
            print()
            sys.exit(1)
    
    if full:
        augment_train_full(batch_start=batch_start, index_start=index_start, append=append)
    else:
        create_augment_pkl()
        augment_from_pkl(num_procs=num_procs)
        
    print('Augmentations saved at {}/data_aug.npz'.format(fileio.WORKING_PREFIX))
    print('AUGMENTATION COMPLETE!')

