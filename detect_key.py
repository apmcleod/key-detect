#!/usr/bin/env python

import sys
import argparse
import logging

import numpy as np
import torch
from sklearn.externals import joblib

# local modules
import fileio
import features
import keys

FS = fileio.FS
NUM_SAMPLES = fileio.NUM_SAMPLES
LOGGING_LEVELS = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}
OUTPUT_PREFIX = fileio.OUTPUT_PREFIX

# Expected input format:
#     Sample rate: 44.1 KHz
#     Sample size: 16 bit
#     Number of channels: 1 (mono)
#     Encoding: WAV


def print_output(output, file=None):
    if file is None:
        print(output)
    else:
        with open(file, 'w') as ff:
            print(output, file=ff)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detects the key of a given "
            "wav file input. Uses only the first 30 seconds of the provided "
            "audio file. The output is of the form: "
            "<tonic {A, A#, Bb, ...}>\t<mode {major, minor}>")
    parser.add_argument("-m", "--model", help="Which model to use for prediction",
                        type=int, choices=range(1, 4), required=True)
    parser.add_argument("-i", "--input", help="Path to input wav", required=True)
    parser.add_argument("-o", "--output", help="Path to output file",
                        default=None)
    parser.add_argument("-v", "--verbosity", help="Level of verbosity (0=errors "
                        "only, 1=some info, 2=most verbose)", type=int, 
                        choices=range(3), default=0)
    
    args = parser.parse_args()
    input_file = args.input
    output_file = args.output
    verbosity = args.verbosity
    model = args.model
    
    logging.basicConfig(level=LOGGING_LEVELS[verbosity])
    
    logging.info("Preprocessing {}".format(input_file))
    logging.debug("Reading file")
    audio_data = fileio.read_audio_data(input_file, FS)
    
    logging.debug("Resampling to {} Hz and clipping to {} samples".
                  format(FS, NUM_SAMPLES))
    audio_data = fileio.cut_or_pad_to_length(audio_data, NUM_SAMPLES)
    
    logging.info("Generating CQT features")
    cqt = features.get_cqt(audio_data)
    
    if model in [1, 2]:
        if model == 1:
            msg = "Using MO1 - new model"
            model_pkl_loc = '{}/MO1_model_best.pkl'.format(OUTPUT_PREFIX)
        if model == 2:
            msg = "Using MO2 - reimplementation of FK1 2017"
            model_pkl_loc = '{}/MO2_model_best.pkl'.format(OUTPUT_PREFIX)
        raise NotImplementedError()
        logging.info(msg)
        logging.info("Instantiating Neural Network")
        net = torch.load(model_pkl_loc).to(device)
        logging.debug("Pytorch model loaded")
        logging.info("Making prediction")
        x_test = torch.from_numpy(cqt[np.newaxis, :]).float()
        x_test = x_test.to(device)
        probs = net(x_test)
        pred = np.argmax(probs, axis=1).squeeze()
        pred_str = keys.STRING_MAP[pred]  
        print_output(pred_str, output_file)
    elif model == 3:
        msg = "Using MO3 - Logistic regression baseline"
        model_pkl_loc = '{}/MO3_model_best.pkl'.format(OUTPUT_PREFIX)
        logging.info(msg)
        logging.debug("Taking mean over time of the cqt")
        x_test = np.mean(cqt, axis=1)
        logging.debug("Subtracting the mean power level (normalising the data)")
        x_test -= np.mean(x_test)
        logging.debug("Loading model")
        mdl = joblib.load(model_pkl_loc)
        logging.debug("Making prediction")
        pred = mdl.predict(x_test[np.newaxis, :])[0]
        pred_str = keys.STRING_MAP[pred]  
        print_output(pred_str, output_file)
    