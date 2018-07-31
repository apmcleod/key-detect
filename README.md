# key-detect

## Data

### GTZAN
GTZAN wav data: http://marsyasweb.appspot.com/download/data_sets/  
GTZAN labels: https://github.com/alexanderlerch/gtzan_key  
(we decided not to use GTZAN keys as labelling method seemed dodgey: http://visal.cs.cityu.edu.hk/downloads/#gtzankeys cite: Tom L.H. Li and Antoni B. Chan, In: Intl. Conference on MultiMedia Modeling (MMM), Taipei, Jan 2011.)  

### GiantSteps
GiantStepsKey (electronic music): https://github.com/GiantSteps/giantsteps-key-dataset

### MSD
Labels from: https://labrosa.ee.columbia.edu/millionsong/
Data from: ...

## Notes
https://www.dropbox.com/s/dnca94fnr5afmgb/Transposition%20mission.pdf?dl=0


## Installation
```
conda create -n key python=3.6 numpy matplotlib jupyter pandas scipy scikit-learn cython seaborn
conda activate key
conda install pytorch torchvision -c pytorch
conda install -c conda-forge librosa
pip install madmom
```
