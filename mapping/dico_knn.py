from sklearn import preprocessing
import numpy as np



def standardize_vects(vectors):

    # standardization of dependent variables
    standard = preprocessing.scale(vectors)
    #print(standard)

    return standard



def get_nearest(v_1, set_others, w_2):

    nearest_word = None
    nearest_idx = None
    min_dist = 999999999999999
    for i, v_2 in enumerate(set_others):
        dist = np.linalg.norm(v_1 - v_2)
        if dist < min_dist:
            min_dist = dist
            nearest_word = w_2[i]
            nearest_idx = i

    return nearest_word, nearest_idx


def get_map(standard_1, standard_2, w_1, w_2, idx_1, idx_2):

    print(len(standard_2))
    print(len(w_2))

    map_words = {}
    map_idx = {}

    for i, v_1 in enumerate(standard_1):
        if i % 1000 == 0:
            print(i)
        nearest_word, nearest_idx = get_nearest(v_1, standard_2, w_2)
        #print(w_1[])
        #print(nearest)

        map_words[w_1[i]] = nearest_word
        map_idx[idx_1[i]] = idx_2[nearest_idx]


    return map_words, map_idx




def get_dico_knn(vectors_1, vectors_2, w_1, w_2, idx_1, idx_2):

    print(vectors_2.shape)

    standard_1 = standardize_vects(vectors_1)
    standard_2 = standardize_vects(vectors_2)

    print("map 1")

    map_1_2, map_idx_1_2 = get_map(standard_1, standard_2, w_1, w_2, idx_1, idx_2)

    print("map 2")
    map_2_1, map_idx_2_1 = get_map(standard_2, standard_1,  w_2,w_1,  idx_2, idx_1)


    return map_1_2, map_2_1, map_idx_1_2, map_idx_2_1