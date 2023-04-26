

from torch import load, norm
from joblib import load as ld
from joblib import dump
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


embeddings = model.encoder.embedding
words_idx = ld(f"../voc/idx_to_word.joblib")
words_freq = ld(f"../voc/word_freq.joblib")

norms = norm(embeddings.weight, dim = 1)



print(norms.shape)


x = []
y = []
for k in range(embeddings.weight.shape[0]):
    word = words_idx[k]
    #print(word)
    y.append(norms[k])
    x.append(words_freq[word])

#print(x)
#print(y)



dump(x, "freqs.joblibs")

dump(y, "norms.joblibs")

















