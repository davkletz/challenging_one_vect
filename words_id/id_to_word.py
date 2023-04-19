from joblib import dump
import sys
path = "/data/mdehouck/thick_vectors/models"


lng = sys.argv[1]


id_to_word = {}
word_to_id = {}
with open(f"{path}/res_k_{1}_seed_{0}_{lng}_gsd.words", "r") as f:
    for line in f:

        vals = line.split(" ")

        if len(vals) != 2:
            print("ERROR")
            print(vals)

            new_vals = [vals[0]+vals[1], vals[2]]
        else:
            if vals[1][-1] == "\n":
                vals[1] = vals[1][:-1]




            id_to_word[int(vals[1])] = vals[0]
            word_to_id[vals[0]] = int(vals[1])


dump(id_to_word, f"/data/dkletz/Other_exp/AvecMatthieu/dicos_ids_words/{lng}_id_to_word.joblib")
dump(word_to_id, f"/data/dkletz/Other_exp/AvecMatthieu/dicos_ids_words/{lng}_word_to_id.joblib")