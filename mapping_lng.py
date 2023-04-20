import sys
from tools.get_vectors import get_vectors
from mapping.dico_knn import get_dico_knn


lng_1 = sys.argv[1]
lng_2 = sys.argv[2]


k = 1
seed = 0


file_1 = f"/data/mdehouck/thick_vectors/models/res_k_{k}_seed_{seed}_{lng_1}_uas"
file_2 = f"/data/mdehouck/thick_vectors/models/res_k_{k}_seed_{seed}_{lng_2}_uas"

vects_1 = get_vectors(file_1, k, seed)

vects_2 = get_vectors(file_2, k, seed)



dico_1_2, dico_2_1 = get_dico_knn(vects_1, vects_2)