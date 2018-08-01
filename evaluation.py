import keys
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def get_scores(targets, guesses):
    """
    Parameters
    ----------
    targets : 1D array
        The list of true keys. Each key is an integer.
    guesses : 1D array
        Like targets
    
    Returns
    -------
    scores : 1D array
        The score for each guess
    """
    nr_obs = targets.shape[0]
    scores = np.array([keys.get_vector_from_key(target) for target in targets])
    scores *= keys.KEY_SUM  # un-normalise
    return scores[np.arange(nr_obs), guesses]


def get_score_single(target, guess):
    scores = keys.get_vector_from_key(target) * keys.KEY_SUM
    return scores[guess]


def plot_confusion_matrix(true, pred, classes=None, title='Confusion matrix', norm=False, **kwargs):
    """Plots a confusion matrix."""
    cm = confusion_matrix(true, pred)
    heatmap_kwargs = dict(annot=True, fmt='d')
    if norm:
        cm_norm = cm/cm.sum(axis=1)[:, np.newaxis]
        heatmap_kwargs['data'] = cm_norm
        heatmap_kwargs['vmin']=0.
        heatmap_kwargs['vmax']=1.
        heatmap_kwargs['fmt']='.3f'
    else:
        heatmap_kwargs['data'] = cm
    if classes is not None:
        heatmap_kwargs['xticklabels']=classes
        heatmap_kwargs['yticklabels']=classes
    heatmap_kwargs.update(kwargs)
    sns.heatmap(**heatmap_kwargs)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')