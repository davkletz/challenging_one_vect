


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
    tot = sum(tens)
    results_norms.append(torch.norm(tens)/len(tens))
    results_freq.append(len(tens))
    results[element] = [torch.norm(tens)/len(tens), len(tens)]


dump(results, "results_990000")





