# Edinburgh_key_detect_1
IDs: MO1, MO2, MO3

## Authors

Andrew McLeod, University of Edinburgh, amcleaod8@staffmail.ed.ac.uk  
James Owers, University of Edinburgh, james.owers@ed.ac.uk  

## Installation

```
conda create -n key python=3.6 numpy pandas scipy scikit-learn h5py
conda activate key
conda install pytorch -c pytorch
conda install -c conda-forge librosa
```

## Executable help

```
: ./detect_key.py -h
usage: detect_key.py [-h] -m {1,2,3} -i INPUT [-o OUTPUT] [-v {0,1,2}]

Detects the key of a given wav file input. Uses only the first 30 seconds of
the provided audio file. The output is of the form: <tonic {A, A#, Bb, ...}>
<mode {major, minor}>

optional arguments:
  -h, --help            show this help message and exit
  -m {1,2,3}, --model {1,2,3}
                        Which model to use for prediction
  -i INPUT, --input INPUT
                        Path to input wav
  -o OUTPUT, --output OUTPUT
                        Path to output file
  -v {0,1,2}, --verbosity {0,1,2}
                        Level of verbosity (0=errors only, 1=some info, 2=most
                        verbose)
```

## Example call

```
conda activate key
input=data/raw/genres/blues/blues.00000.au  # input data (can be wav/mp3/au)
output=blues.00000.au.prediction.txt        # file to write precition to
verbosity=2
model=1  # 1: MO1, 2: MO2, 3: MO3

./detect_key.py -m $model -i $input -o $output -v $verbosity
```

Another example:
```
./detect_key.py --model 1 \
                --input data/raw/genres/blues/blues.00000.au \
                # --output out.txt \  # excluding output arg prints to stdout
                # --verbosity 0  # 0 gives minimal printing to console (default)
```

## Additional information

* Number of threads/cores used: 1
* Expected memory footprint: < 256 MB
* Expected runtime: < 5 seconds
* Requirements: latest version of [Conda](https://conda.io/docs/user-guide/install/download.html)


