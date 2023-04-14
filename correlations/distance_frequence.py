from sklearn.neighbors.nearest_centroid import NearestCentroid
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


def get_cluster_centroid(X, y):


    clf = NearestCentroid()
    clf.fit(X, y)

    return clf.centroids_



k = int(sys.argv[1])

word = sys.argv[2]

nb_ex = int(sys.argv[3])

lng = "fr"

try:
    seed = sys.argv[4]
except:
    seed = "0"


model_name = f"/data/mdehouck/thick_vectors/models/res_k_{k}_seed_{seed}_fr_gsd_uas"
device = "cpu"
model = torch.load(model_name, map_location=device)


list_vectors = model["W.weight"]

list_vectors = list_vectors.cpu().numpy()

list_vectors = get_list_vectors(list_vectors, k)


labs = [0 for i in list_vectors]

cluster_centroid = get_cluster_centroid(list_vectors, labs)
