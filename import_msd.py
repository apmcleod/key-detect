#!/usr/bin/env python

import fileio
fileio.load_all_data(datasets=['msd'], chunk_start_nr=32)
fileio.make_labels()
fileio.train_test_split()
fileio.make_Y()