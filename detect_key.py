#!/usr/bin/env python
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detects the key of a given "
            "wav file input. Uses only the first 30 seconds of the provided "
            "audio file. The output is of the form:\n\t"
            "<tonic {A, A#, Bb, ...}>\t<mode {major, minor}>\n")
    parser.add_argument("-i", "--input", help="Path to input file for key "
                        "detection")
    parser.add_argument("-o", "--output", help="Path to output file for "
                        "the result")
    print("Preprocessing input file")
    print("Resampling and clipping input file")
    print("Generating CQT features")
    # Sample rate: 44.1 KHz
    # Sample size: 16 bit
    # Number of channels: 1 (mono)
    # Encoding: WAV

    print("Instantiating Neural Network")

    print("Making prediction")

    print("DONE!")

