from torch import load



def get_list_vectors(list_vectors, k):

    results = []
    size_vectors = list_vectors.shape[-1]
    size_real_vectors = size_vectors // k

    for i in range(k):
        results.append(list_vectors[:, i*size_real_vectors:(i+1)*size_real_vectors])

    return results

def get_vectors(name_corp, k, seed):
    model_name = f"/data/mdehouck/thick_vectors/models/res_k_{k}_seed_{seed}_{name_corp}_uas"
    device = "cpu"
    model = load(model_name, map_location=device)




    list_vectors = model["W.weight"]

    list_vectors = list_vectors.cpu().numpy()
    #print(list_vectors.shape)

    list_vectors = get_list_vectors(list_vectors, k)

    return list_vectors