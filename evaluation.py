import keys
import numpy as np


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