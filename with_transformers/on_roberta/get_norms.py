


import torch
from joblib import dump
a = torch.load("/data/mnedeljkovic/thesis/thesis/code/embeddings/embeddings990000")

print("loaded")
b = list(a.keys())


results = {}
results_norms = []
results_freq = []
for element in b:
    current_list = a[element]
    tens = a[b][1]

    tot = sum(tens)
    results_norms.append(torch.norm(tens)/len(tens))
    results_freq.append(len(tens))
    results[b] = [torch.norm(tens)/len(tens), len(tens)]


dump(results, "results_990000")





