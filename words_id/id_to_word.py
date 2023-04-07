from joblib import dump

path = "/data/mdehouck/thick_vectors/models"

k = "4"
lng = "fr"
seed = "0"

id_to_word = {}
word_to_id = {}
with open(f"{path}/res_k_{k}_seed_{seed}_{lng}_gsd.words", "r") as f:
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


dump(id_to_word, f"/data/dkletz/Other_exp/AvecMatthieu/dicos_ids_words/res_k_{k}_seed_{seed}_{lng}_gsd_id_to_word.joblib")
dump(word_to_id, f"/data/dkletz/Other_exp/AvecMatthieu/dicos_ids_words/res_k_{k}_seed_{seed}_{lng}_gsd_word_to_id.joblib")