import sys
from tools.get_vectors import get_vectors
from mapping.dico_knn import get_dico_knn
from joblib import load, dump
from mapping.norm_freq_set import get_norm_freq_sets, get_freq_sets

lng_1 = sys.argv[1]
corpus_1 = sys.argv[2]
lng_2 = sys.argv[3]
corpus_2 = sys.argv[4]

try:
    norm_freq = sys.argv[5]
    if norm_freq in ["True", "true", "1", "y", "yes", "Yes", "Y", "oui", "Oui", "O", "o"]:
        norm_freq = True
    else:
        norm_freq = False

except:
    norm_freq = True

k = 1
seed = 0


#file_1 = f"/data/mdehouck/thick_vectors/models/res_k_{k}_seed_{seed}_{lng_1}_uas"
#file_2 = f"/data/mdehouck/thick_vectors/models/res_k_{k}_seed_{seed}_{lng_2}_uas"

vects_1 = get_vectors(lng_1, k, seed)
vects_2 = get_vectors(lng_2, k, seed)
#print((vects_2))

id_to_word_1 = load(f"/data/dkletz/Other_exp/AvecMatthieu/dicos_ids_words/{lng_1[:2]}_id_to_word.joblib")
id_to_word_2 = load(f"/data/dkletz/Other_exp/AvecMatthieu/dicos_ids_words/{lng_2[:2]}_id_to_word.joblib")

#print(id_to_word_1)


path = "tools"

dico_freq_1 = load(f"{path}/dico_{corpus_1}_{lng_1}-ud-train.joblib")
dico_freq_2 = load(f"{path}/dico_{corpus_2}_{lng_2}-ud-train.joblib")


for i in range(k):

    if norm_freq:
        vects_1, w_1, idx_1 = get_norm_freq_sets(vects_1[i], id_to_word_1, dico_freq_1)
        vects_2, w_2, idx_2 = get_norm_freq_sets(vects_2[i], id_to_word_2, dico_freq_2)
        path = "norm_freq"
    else:
        vects_1, w_1, idx_1 = get_freq_sets(vects_1[i], id_to_word_1, dico_freq_1)
        vects_2, w_2, idx_2 = get_freq_sets(vects_2[i], id_to_word_2, dico_freq_2)
        path = "freq"

    dico_1_2, dico_2_1, map_idx_1_2, map_idx_2_1 = get_dico_knn(vects_1, vects_2, w_1, w_2, idx_1, idx_2)

    dump(dico_1_2, f"/data/dkletz/Other_exp/AvecMatthieu/dicos_knn/{path}/{lng_1}_{lng_2}_{corpus_1}_{corpus_2}_k_{k}_seed_{seed}.joblib")
    dump(dico_2_1, f"/data/dkletz/Other_exp/AvecMatthieu/dicos_knn/{path}/{lng_2}_{lng_1}_{corpus_2}_{corpus_1}_k_{k}_seed_{seed}.joblib")
    dump(map_idx_1_2,
         f"/data/dkletz/Other_exp/AvecMatthieu/dicos_knn/{path}/{lng_1}_{lng_2}_{corpus_1}_{corpus_2}_k_{k}_seed_{seed}_map_idx_1_2.joblib")

    dump(map_idx_2_1,
         f"/data/dkletz/Other_exp/AvecMatthieu/dicos_knn/{path}/{lng_2}_{lng_1}_{corpus_2}_{corpus_1}_k_{k}_seed_{seed}_map_idx_2_1.joblib")



