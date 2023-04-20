import numpy
import numpy as np
from joblib import load







def get_norm_freq_vect(vect, i, id_to_word, dico_freq):

    norm_vect = np.linalg.norm(vect)
    freq = dico_freq[id_to_word[i]]

    return [norm_vect, freq]


def get_norm_freq_sets(set_vects, id_to_word, dico_freq):
    """
    :param set_vects: list of list of vectors
    :return: list of list of normalized frequencies
    """
    norm_freq_sets = []
    for i, vector in enumerate(set_vects):
        norm_freq_sets.append(get_norm_freq_vect(vector, i, id_to_word, dico_freq))

    norm_freq_sets = np.array(norm_freq_sets)

    return norm_freq_sets