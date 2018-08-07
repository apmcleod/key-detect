#!/usr/bin/env python
import argparse
import logging
import sys
import fileio
import features
import torch

FS = fileio.FS
NUM_SAMPLES = fileio.NUM_SAMPLES
LOGGING_LEVELS = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}

# Expected input format:
#     Sample rate: 44.1 KHz
#     Sample size: 16 bit
#     Number of channels: 1 (mono)
#     Encoding: WAV

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detects the key of a given "
            "wav file input. Uses only the first 30 seconds of the provided "
            "audio file. The output is of the form: "
            "<tonic {A, A#, Bb, ...}>\t<mode {major, minor}>")
    parser.add_argument("-i", "--input", help="Path to input wav", required=True)
    parser.add_argument("-o", "--output", help="Path to output file", default=sys.stdout)
    parser.add_argument("-v", "--verbosity", help="Level of verbosity (0=None, 2=most verbose)",
                        type=int, choices=range(3), default=0)
    args = parser.parse_args()
    input_file = args.input
    output_file = args.output
    verbosity = args.verbosity
    
    logging.basicConfig(level=LOGGING_LEVELS[verbosity])
    
    logging.info("Preprocessing {}".format(input_file))
    logging.debug("\tReading file")
    audio_data = fileio.read_audio_data(input_file, FS)
    
    logging.debug("\tResampling to {} Hz and clipping to {} samples".format(FS, NUM_SAMPLES))
    audio_data = fileio.cut_or_pad_to_length(audio_data, NUM_SAMPLES)
    
    logging.info("Generating CQT features")
    cqt = features.get_cqt(song)
    
    logging.info("Instantiating Neural Network")
    net = 
    
    logging.debug("Pytorch model loaded")
    
    
    logging.info("Making prediction")