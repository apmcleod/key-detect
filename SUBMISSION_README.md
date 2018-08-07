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
./detect_key --method [1,2,3] --input file --output file
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


