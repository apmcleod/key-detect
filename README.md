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

### Packages
```
conda create -n key python=3.6 numpy matplotlib jupyter pandas scipy scikit-learn cython seaborn
conda activate key
conda install pytorch torchvision -c pytorch
conda install -c conda-forge librosa
pip install madmom
```

### Data Download
```
# creates ./genres - the gtzan data
cd data/raw
wget http://opihi.cs.uvic.ca/sound/genres.tar.gz
tar -xzf genres.tar.gz

# creates ./gtzan_key-master - the gtzan labels
wget https://github.com/alexanderlerch/gtzan_key/archive/master.zip
unzip master.zip

# creates ./giantsteps-key-dataset/audio and ./giantsteps-key-dataset/annotations/keys
#     the data and labels for giantsteps 
wget https://github.com/GiantSteps/giantsteps-key-dataset/archive/master.zip
unzip master.zip
cd giantsteps-key-dataset-master
./audio_dl.sh

# creates MillionSongSubset/data - the Million song dataset labels
wget http://static.echonest.com/millionsongsubset_full.tar.gz
tar -xzf millionsongsubset_full.tar.gz
# creates MillionSongSubset
```

### Preprocessing pipeline
```
conda activate key
./fileio.py
```