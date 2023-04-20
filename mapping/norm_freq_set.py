import numpy
import numpy as np
from joblib import load







def get_norm_freq_vect(vect, i, id_to_word, dico_freq):

    norm_vect = np.linalg.norm(vect)
    if i not in id_to_word:
        print(f"i {i} not in id_to_word")
        return None
    word = id_to_word[i]
    if word in dico_freq:
        freq = dico_freq[word]
        return [norm_vect, freq]
    print(f"word {word} not in dico_freq")
    return None


def get_norm_freq_sets(set_vects, id_to_word, dico_freq):
    """
    :param set_vects: list of list of vectors
    :return: list of list of normalized frequencies
    """
    norm_freq_sets = []
    for i, vector in enumerate(set_vects):
        q_v = get_norm_freq_vect(vector, i, id_to_word, dico_freq)
        if q_v is not None:
            norm_freq_sets.append(q_v)


    norm_freq_sets = np.array(norm_freq_sets)

    return norm_freq_sets