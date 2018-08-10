from IPython import display
import numpy as np
import matplotlib.pyplot as plt


def in_ipynb():
    try:
        get_ipython
        return True
    except NameError:
        return False


def play_from_label(label):
    chunk_loc = './data/working/chunks'
    fs = 22050
    chunk_nr = label['chunk_nr']
    X = np.load('{}/chunk{:04d}.npz'.format(chunk_loc, int(chunk_nr)))['X']
    idx = int(label['chunk_idx'])
    if in_ipynb():
        display.display(display.Audio(X[idx], rate=fs))
    else:
        raise NotImplementedError


def grouped_bar_chart(array, legend_labels=None, bar_labels=None,
                      border_size=0.1, legend=True):
    """
    Plots a grouped bar chart where each row of input array contains the
    values of consecutive bars.
    """
    nr_obs, nr_bars = array.shape
    pos = np.arange(nr_bars)
    if legend_labels is None:
        legend_labels = np.arange(nr_obs)
    if bar_labels is None:
        bar_labels = np.arange(nr_bars)
    bar_width = (1-border_size)/nr_obs
    centre_offset = (nr_obs/2 - .5)*bar_width
    for ii in range(nr_obs):
        offset = bar_width*ii - centre_offset
        plt.bar(pos+offset, array[ii], bar_width, label=legend_labels[ii])
    plt.xticks(pos, bar_labels)
    if legend:
        plt.legend()
        

def number_countplot(ax=None):
    if ax is None:
        ax = plt.gca()
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:d}'.format(height),
                ha="center")
        

def plot_train_test_log(filename):
    """Simple function for plotting a csv of the form:
    col,header,here
    0,1,2
    3,4,5
    4,3,2
    1,0,-1
    """
    pd.read_csv(filename).plot()
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    