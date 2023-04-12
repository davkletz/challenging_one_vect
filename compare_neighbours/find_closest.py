import torch.nn as nn
import torch
import numpy as np
from random import shuffle
from joblib import load
import sys
cos = nn.CosineSimilarity(dim = 0)



def most_similars( idx_vect_to_compare, all_list_vectors, n_k):

    closests = []


    k = 0

    for list_vectors in all_list_vectors:
        vect_to_compare = list_vectors[idx_vect_to_compare]
        #size_vector = vect_to_compare.shape[-1]
        current_vect_to_compare = vect_to_compare

        list_similarities = []


        for element in list_vectors:
            current_cos = cos(current_vect_to_compare, element)

            list_similarities.append(current_cos)

        #print(list_similarities)

        list_similarities = torch.tensor(list_similarities)

        sorted, indices = torch.sort(list_similarities)

        indices = indices[-n_k:].cpu().numpy()

        indices = np.flip(indices)

        closests.append(indices)




    return closests


def get_list_vectors(list_vectors, k):

    results = []
    size_vectors = list_vectors.shape[-1]
    size_real_vectors = size_vectors // k

    for i in range(k):
        results.append(list_vectors[:, i*size_real_vectors:(i+1)*size_real_vectors])

    return results



k = int(sys.argv[1])

lng = "fr"
seed = "0"
model_name = f"/data/mdehouck/thick_vectors/models/res_k_{k}_seed_0_fr_gsd_uas"
device = "cpu"
model = torch.load(model_name, map_location=device)


list_vectors = model["W.weight"]

list_vectors = get_list_vectors(list_vectors, k)

r = most_similars(0, list_vectors, 25)

#print(r)


id_to_word = load(f"/data/dkletz/Other_exp/AvecMatthieu/dicos_ids_words/{lng}_gsd_id_to_word.joblib")
word_to_id = load(f"/data/dkletz/Other_exp/AvecMatthieu/dicos_ids_words/{lng}_gsd_word_to_id.joblib")

gr = word_to_id["grand"]

r = most_similars(gr, list_vectors, 25)

for results in r:
    print(f'\n\n###')
    for i, element in enumerate(results):
        if element in id_to_word:
            print(id_to_word[element])
        else:
            print(f"not in dico : {element}")

list_idx = list(range(len(list_vectors[0])))

shuffle(list_idx)

list_idx = list_idx[:15]



"""
for l in list_idx:

    print(f"\n\n#####\n\n{id_to_word[l]}\n\n")
    r = most_similars(l, list_vectors, 25)



    dico_f = {}

    for results in r:
        print(f'\n\n###')
        for i, element in enumerate(results):
            if element in id_to_word:
                print(id_to_word[element])
            else:
                print(f"not in dico : {element}")
            if element in dico_f:
                dico_f[element].append(i)
            else:
                dico_f[element] = [i]

    for j in dico_f:
        if len(dico_f[j]) > 1:
            print(f"{j} : {dico_f[j]}")





#for l in range(len(list_vectors)):

"""