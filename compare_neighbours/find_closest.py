import torch.nn as nn
import torch

cos = nn.CosineSimilarity(dim = 0)



def most_similar( vect_to_compare, list_vectors, n_k):

    list_similarities = []

    for element in list_vectors:
        list_similarities.append(cos(vect_to_compare, element))

    list_similarities = torch.tensor(list_similarities)

    sorted, indices = torch.sort(list_similarities)

    return indices[-n_k:]


def get_list_vectors(list_vectors, k):

    results = []
    size_vectors = list_vectors.shape[-1]
    size_real_vectors = size_vectors // k

    for i in range(k):
        results.append(list_vectors[:, i*size_real_vectors:(i+1)*size_real_vectors])

    return results



k = 4
model_name = f"/data/mdehouck/thick_vectors/models/res_k_{k}_seed_0_fr_gsd_uas"
device = "cpu"
model = torch.load(model_name, map_location=device)


list_vectors = model["W.weight"]

list_vectors = get_list_vectors(list_vectors)

most_similar(list_vectors[0], list_vectors, 10)