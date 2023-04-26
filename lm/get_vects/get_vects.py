

from torch import load


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


embeddings = model.encoder.embeddings