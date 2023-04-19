from io import open
from conllu import parse_incr
from joblib import dump


def get_freq(path):

    dico = {}

    data_file = open(path,"r", encoding="utf-8")
    for tokenlist in parse_incr(data_file):
        for token_avail in tokenlist:
            print(token_avail['form'])
            if token_avail['form'] in dico:
                dico[token_avail['form']] += 1
            else:
                dico[token_avail['form']] = 1


    return dico





if __name__ == "__main__":

    path = "/data/mdehouck/UD_French-GSD/fr_gsd-ud-train.conllu"
    dico = get_freq(path)
    print(dico)