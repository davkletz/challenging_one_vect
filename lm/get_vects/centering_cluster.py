from sklearn.cluster import KMeans
from joblib import dump
from torch import load
from numpy.linalg import norm as norm
from joblib import load as ld


def centering_cluster(X, n_clusters=1, random_state=0):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")

    kmeans.fit(X)

    centroid = kmeans.cluster_centers_

    centered_X = X - centroid


    return centered_X


def get_vects(path, device):
    """
    Get the vectors from the language model
    :param path: path to the language model
    :param device: device to use
    :return: the vectors
    """
    model = load(path, map_location=device)
    return model.encoder.weight.data




path = "/data/dkletz/Other_exp/AvecMatthieu/LSTM_ambiguity/language_model"
file = "model.pt"

model = load(f"{path}/{file}")


embeddings = model.encoder.embedding.weight.detach().cpu().numpy()


centered_cluster  = centering_cluster(embeddings, n_clusters=1, random_state=0)


norms = norm(centered_cluster, dim = 1)

#norms = norms.detach().cpu().numpy()


words_idx = ld(f"../voc/idx_to_word.joblib")
words_freq = ld(f"../voc/word_freq.joblib")



y = []
for k in range(embeddings.weight.shape[0]):
    word = words_idx[k]
    #print(word)
    y.append(norms[k])


dump(y, path + "centroids_norms.joblib")

