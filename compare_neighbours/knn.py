import torch
import numpy as np
from random import shuffle
from joblib import load
import sys
from sklearn.neighbors import NearestNeighbors
from nltk.corpus import stopwords
french_stopwords = set(stopwords.words('french'))



def eucl_distance(v_1, v_2):
    return torch.norm(v_1 - v_2)

def most_similars( idx_vect_to_compare, all_list_vectors, n_k):

    closests = []
    distances = []


    k = 0

    for list_vectors in all_list_vectors:
        vect_to_compare = list_vectors[idx_vect_to_compare]
        #size_vector = vect_to_compare.shape[-1]
        current_vect_to_compare = vect_to_compare

        list_similarities = []


        for element in list_vectors:
            current_dist = eucl_distance(current_vect_to_compare, element)

            list_similarities.append(current_dist)

        #print(list_similarities)

        list_similarities = torch.tensor(list_similarities)

        sorted, indices = torch.sort(list_similarities)
        #print(sorted)

        indices = indices[:n_k].cpu().numpy()
        sorted = sorted[:n_k].cpu().numpy()

        closests.append(indices)
        distances.append(sorted)




    return closests, distances


def get_list_vectors(list_vectors, k):

    results = []
    size_vectors = list_vectors.shape[-1]
    size_real_vectors = size_vectors // k

    for i in range(k):
        results.append(list_vectors[:, i*size_real_vectors:(i+1)*size_real_vectors])

    return results



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

list_vectors = get_list_vectors(list_vectors, k)

#r = most_similars(0, list_vectors, nb_ex)

#print(r)


id_to_word = load(f"/data/dkletz/Other_exp/AvecMatthieu/dicos_ids_words/{lng}_gsd_id_to_word.joblib")
word_to_id = load(f"/data/dkletz/Other_exp/AvecMatthieu/dicos_ids_words/{lng}_gsd_word_to_id.joblib")

gr = word_to_id[word]

r, d = most_similars(gr, list_vectors,nb_ex)

for j, results in enumerate(r):
    print(f'\n\n###')
    for i, element in enumerate(results):
        if element in id_to_word:
            if id_to_word[element] not in french_stopwords:
                print(f"{id_to_word[element]} : {d[j][i]}" )
        else:
            print(f"not in dico : {element} : {d[j][i]}")

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