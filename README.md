# key-detect

## Data

* GTZAN
    GTZAN wav data: http://marsyasweb.appspot.com/download/data_sets/  
    GTZAN labels: https://github.com/alexanderlerch/gtzan_key  
    (we decided not to use GTZAN keys as labelling method seemed dodgey: 
    http://visal.cs.cityu.edu.hk/downloads/#gtzankeys cite: Tom L.H. Li 
    and Antoni B. Chan, In: Intl. Conference on MultiMedia Modeling (MMM), 
    Taipei, Jan 2011.)  
* GiantSteps
    GiantStepsKey (electronic music): https://github.com/GiantSteps/giantsteps-key-dataset
* GiantSteps-MTG
    GiantStepsKey 2 (electronic music): https://github.com/GiantSteps/giantsteps-mtg-key-dataset
* MSD
    Labels from: https://labrosa.ee.columbia.edu/millionsong/
    Data from: please contact authors fmi

## Installation

### Packages
```
conda create -n key python=3.6 numpy matplotlib jupyter pandas scipy scikit-learn cython seaborn h5py
conda activate key
conda install pytorch torchvision -c pytorch
conda install -c conda-forge librosa
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
rm master.zip

# creates ./giantsteps-key-dataset/audio and ./giantsteps-key-dataset/annotations/keys
#     the data and labels for giantsteps 
wget https://github.com/GiantSteps/giantsteps-key-dataset/archive/master.zip
unzip master.zip
cd giantsteps-key-dataset-master
./audio_dl.sh
rm master.zip

# creates ./giantsteps-mtg-key-dataset/audio and ./giantsteps-mtg-key-dataset/annotations/keys
#     the data and labels for giantsteps 
wget https://github.com/GiantSteps/giantsteps-mtg-key-dataset/archive/master.zip
unzip master.zip
cd giantsteps-mtg-key-dataset-master
./audio_dl.sh
rm master.zip

# creates MillionSongSubset/data - the Million song dataset labels
# please email james.owers@ed.ac.uk for instructions information about MSD .mp3 files
wget http://static.echonest.com/millionsongsubset_full.tar.gz
tar -xzf millionsongsubset_full.tar.gz
# creates MillionSongSubset
```

### Preprocessing pipeline
```
conda activate key
./fileio.py
./features.py
./data_aug.py  # creates augmented training data by running fileio and features methods. Outputs
                 # data_aug.npz containing X_aug and Y_aug (based on splits.npz in and labels.pkl)
```

### Modelling
When modelling you can now begin in this way:
```
DATA_DIR = './data/working'

labels = pd.read_pickle("{}/labels.pkl".format(DATA_DIR))

with np.load("{}/splits.npz".format(DATA_DIR)) as splits:
    train_idx = splits['train_idx']
    test_idx = splits['test_idx']

X = np.load("{}/X_cqt.npz".format(DATA_DIR))['X']
X_train = X[train_idx, :]
X_test = X[test_idx, :]

Y = np.load("{}/Y.npz".format(DATA_DIR))['Y']
Y_train = Y[train_idx, :]
Y_test = Y[test_idx, :]

with np.load("{}/data_aug.npz".format(DATA_DIR)) as data:
    X_aug = data['X']
    Y_aug = data['Y']
X_train = np.vstack(X_train, X_aug)
Y_train = np.vstack(Y_train, Y_aug)
```


## Notes
https://www.dropbox.com/s/dnca94fnr5afmgb/Transposition%20mission.pdf?dl=0
