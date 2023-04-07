

path = "/data/mdehouck/thick_vectors/models"

k = "4"

with open(f"{path}/res_k_{k}_seed_0_fr_gsd.words", "r") as f:
    for line in f:
        print(line.split(" "))