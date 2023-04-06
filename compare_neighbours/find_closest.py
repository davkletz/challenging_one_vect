import torch.nn as nn
import torch

cos = nn.CosineSimilarity(dim = 0)



def most_similars( vect_to_compare, all_list_vectors, n_k):

    closests = []

    size_vector = vect_to_compare.shape[-1]

    k = 0

    for list_vectors in all_list_vectors:
        current_vect_to_compare = vect_to_compare[k*size_vector:(k+1)*size_vector]

        list_similarities = []

        for element in list_vectors:
            list_similarities.append(cos(current_vect_to_compare, element))

        print(list_similarities)

        list_similarities = torch.tensor(list_similarities)

        sorted, indices = torch.sort(list_similarities)

        closests.append(indices[-n_k:])

    return closests


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

list_vectors = get_list_vectors(list_vectors, k)

r = most_similars(list_vectors[0], list_vectors, 10)

print(r)

#for l in range(len(list_vectors)):

