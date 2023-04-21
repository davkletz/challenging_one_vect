from sklearn import preprocessing
import numpy as np



def standardize_vects(vectors):

    # standardization of dependent variables
    standard = preprocessing.scale(vectors)
    #print(standard)

    return standard



def get_nearest(v_1, set_others, w_2):

    nearest = None
    min_dist = 999999999999999
    for i, v_2 in enumerate(set_others):
        dist = np.linalg.norm(v_1 - v_2)
        if dist < min_dist:
            min_dist = dist
            nearest = w_2[i]

    return nearest


def get_map(standard_1, standard_2, w_1, w_2):

    map = {}
    for i, v_1 in enumerate(standard_1):
        if i % 1000 == 0:
            print(i)
        nearest = get_nearest(v_1, standard_2, w_2)
        #print(w_1[])
        #print(nearest)

        map[w_1[i]] = nearest


    return map



def get_dico_knn(vectors_1, vectors_2, w_1, w_2):

    print(vectors_2.shape)

    standard_1 = standardize_vects(vectors_1)
    standard_2 = standardize_vects(vectors_2)

    print("map 1")

    map_1_2 = get_map(standard_1, standard_2, w_1, w_2)

    print("map 2")
    map_2_1 = get_map(standard_2, standard_1, w_2, w_1)


    return map_1_2, map_2_1