# Edinburgh_key_detect_1
IDs: MO1, MO2, MO3

## Authors
Andrew McLeod, amcleaod8@staffmail.ed.ac.uk
James Owers, james.owers@ed.ac.uk

## Installation

```
conda create -n key python=3.6 numpy matplotlib jupyter pandas scipy scikit-learn cython seaborn h5py
conda activate key
conda install pytorch torchvision -c pytorch
conda install -c conda-forge librosa
```

## Example call
```
conda activate key
input=data/raw/genres/blues/blues.00000.au  # input data (can be wav/mp3/au)
output=blues.00000.au.prediction.txt        # file to write precition to
verbosity=2
model=1  # 1: MO1, 2: MO2, 3: MO3
./detect_key.py -h
./detect_key.py -m $model -i $input -o $output -v $verbosity
./detect_key.py --model $model \
                --input $input \
                # --output $output \  # excluding output arg prints to stdout
                --verbosity 0  # 0 gives minimal printing to console
```





optional flag specifying which GPU to use to predict? (depend on time diff)
optional call method taking in folders instead of files?

####
Command line calling format for all executables including examples
Number of threads/cores used or whether this should be specified on the command line
Expected memory footprint
Expected runtime
Approximately how much scratch disk space will the submission need to store any feature/cache files?
Any required environments/architectures (and versions) such as Matlab, Java, Python, Bash, Ruby etc.
Any special notice regarding to running your algorithm


