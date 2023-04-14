from sklearn.neighbors import NearestCentroid
import torch
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




k = int(sys.argv[1])


lng = "fr"

try:
    seed = sys.argv[2]
except:
    seed = "0"


model_name = f"/data/mdehouck/thick_vectors/models/res_k_{k}_seed_{seed}_fr_gsd_uas"
device = "cpu"
model = torch.load(model_name, map_location=device)


list_vectors = model["W.weight"]

list_vectors = list_vectors.cpu().numpy()

list_vectors = get_list_vectors(list_vectors, k)




for k in range(len(list_vectors)):
    k_list_vectors = list_vectors[k]
    labs = [0 for i in k_list_vectors]

    cluster_centroid = get_cluster_centroid(k_list_vectors, labs)

    print(cluster_centroid)
    print(cluster_centroid.shape)


