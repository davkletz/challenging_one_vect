


import torch
from joblib import dump
a = torch.load("/data/mnedeljkovic/thesis/thesis/code/embeddings/embeddings990000")

print("loaded")
b = list(a.keys())


results = {}
results_norms = []
results_freq = []
i = 0
for element in b:
    i+=1
    if i%100 == 0:
        print(i)
    current_list = a[element]

    tens = a[element][1]
    if len(tens) == 0:
        results[element] = [0, 0]

    else:
        tot_norms = 0
        for embed in tens:
            current_norm = torch.norm(embed)
            tot_norms += current_norm



        results[element] = [tot_norms/len(tens), len(tens)]


dump(results, "results_averga_norms990000")





