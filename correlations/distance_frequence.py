from sklearn.neighbors import NearestCentroid
from torch import load
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
def most_similars(vect_to_compare, list_vectors, n_k):

    closests = []
    distances = []


    k = 0


    current_vect_to_compare = vect_to_compare

    list_similarities = []

    for element in list_vectors:
        current_dist = eucl_distance(current_vect_to_compare, element)
        list_similarities.append(current_dist)


    list_similarities = np.array(list_similarities)

    sorted = np.sort(list_similarities)
    indices = np.argsort(list_similarities)


    indices = indices[:n_k].cpu().numpy()
    sorted = sorted[:n_k].cpu().numpy()

  




    return indices, sorted


lng = "fr"
id_to_word = load(f"/data/dkletz/Other_exp/AvecMatthieu/dicos_ids_words/{lng}_gsd_id_to_word.joblib")
word_to_id = load(f"/data/dkletz/Other_exp/AvecMatthieu/dicos_ids_words/{lng}_gsd_word_to_id.joblib")



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




for k in range(len(list_vectors)):
    k_list_vectors = list_vectors[k]
    labs = [0 for i in k_list_vectors]

    cluster_centroid = get_cluster_centroid(k_list_vectors)

    print(cluster_centroid)
    print(cluster_centroid.shape)


    update_vects = np.subtract(cluster_centroid, k_list_vectors[0])


    origin = np.zeros(cluster_centroid.shape)

    closest = most_similars(origin, update_vects, 25)












