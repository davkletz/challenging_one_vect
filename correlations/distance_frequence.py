from sklearn.neighbors import NearestCentroid
from torch import load
from joblib import load as ld
from joblib import dump
import numpy as np
import sys



def get_list_vectors(list_vectors, k):

    results = []
    size_vectors = list_vectors.shape[-1]
    size_real_vectors = size_vectors // k

    for i in range(k):
        results.append(list_vectors[:, i*size_real_vectors:(i+1)*size_real_vectors])

    return results

def get_cluster_centroid(arr):
    length = arr.shape[0]
    print(arr.shape)
    sum_ar = np.sum(arr, axis=0)
    print(sum_ar.shape)
    return sum_ar/length


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


def compare_freq_dist(words_id, freqs, dist):
    x = [] #abs : frequence apparition
    y = [] #ordo : distance au centre

    for j, results in enumerate(words_id):
        if results in id_to_word:
            if id_to_word[results] in freqs:
                x.append(freqs[id_to_word[results]])
                y.append(dist[j])

    return x, y


lng = "fr"
id_to_word = ld(f"/data/dkletz/Other_exp/AvecMatthieu/dicos_ids_words/{lng}_gsd_id_to_word.joblib")
word_to_id = ld(f"/data/dkletz/Other_exp/AvecMatthieu/dicos_ids_words/{lng}_gsd_word_to_id.joblib")



k = int(sys.argv[1])


lng = "fr"

try:
    seed = sys.argv[2]
except:
    seed = "0"


model_name = f"/data/mdehouck/thick_vectors/models/res_k_{k}_seed_{seed}_fr_gsd_uas"
device = "cpu"
model = load(model_name, map_location=device)


list_vectors = model["W.weight"]

list_vectors = list_vectors.cpu().numpy()

list_vectors = get_list_vectors(list_vectors, k)

n_k = int(sys.argv[3])


path = "/data/dkletz/Other_exp/AvecMatthieu/challenging_one_vect/tools"
corpus = "UD_French-GSD"
file = "fr_gsd-ud-dev.conllu"
dico_freq = ld(f"{path}/dico_{corpus}_{file[:-7]}.joblib")



for k in range(len(list_vectors)):
    k_list_vectors = list_vectors[k]
    labs = [0 for i in k_list_vectors]

    cluster_centroid = get_cluster_centroid(k_list_vectors)


    update_vects = np.subtract(k_list_vectors, cluster_centroid )
    #update_vects = k_list_vectors


    origin = np.zeros(cluster_centroid.shape)

    indices, sorted = most_similars(origin, update_vects)

    for j, results in enumerate(indices[:n_k]):
        print(f'\n###')
        if results in id_to_word:
            print(f"{id_to_word[results]} : {sorted[j]}")
        else:
            print(f"not in dico : {results} : {sorted[j]}")


    min_dist = np.min(sorted)

    for i in range(len(sorted)):
        sorted[i] = sorted[i]/min_dist

    x, y = compare_freq_dist(indices, dico_freq, sorted)

    print(len(x), len(y))

    dump([x,y], f"freq_dis_fr.joblib")























