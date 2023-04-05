import torch.nn as nn
import torch

cos = nn.CosineSimilarity()



def most_similar( vect_to_compare, list_vectors, n_k):

    list_similarities = []

    for element in list_vectors:
        list_similarities.append(cos(vect_to_compare, element))

    list_similarities = torch.tensor(list_similarities)

    sorted, indices = torch.sort(list_similarities)

    return indices[-n_k:]



model_name = "res_k_4_seed_0_fr_gsd_uas"
device = "cpu"
model = torch.load(model_name, map_location=device)


list_vectors = model["W.weight"]

most_similar(list_vectors[0], list_vectors, 10)