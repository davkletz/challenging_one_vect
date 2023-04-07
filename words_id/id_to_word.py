

path = "/data/mdehouck/thick_vectors/models"

k = "4"



id_to_word = {}
word_to_id = {}
with open(f"{path}/res_k_{k}_seed_0_fr_gsd.words", "r") as f:
    for line in f:

        vals = line.split(" ")

        if len(vals) != 2:
            print("ERROR")
            print(vals)
        else:
            if vals[1][-1] == "\n":
                vals[1] = vals[1][:-1]


            id_to_word[int(vals[1])] = vals[0]
            word_to_id[vals[0]] = int(vals[1])
