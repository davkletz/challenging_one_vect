import numpy as np


def eucl_distance(v_1, v_2):
    return np.linalg.norm(v_1 - v_2)
def most_similars(vect_to_compare, list_vectors):

    current_vect_to_compare = vect_to_compare

    list_similarities = []

    for element in list_vectors:
        current_dist = eucl_distance(current_vect_to_compare, element)
        list_similarities.append(current_dist)


    list_similarities = np.array(list_similarities)

    sorted = np.sort(list_similarities)
    indices = np.argsort(list_similarities)



    return indices, sorted




