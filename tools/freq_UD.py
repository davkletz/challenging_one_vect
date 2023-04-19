from io import open
from conllu import parse_incr
from joblib import dump


def get_freq(path):

    dico = {}

    data_file = open(path,"r", encoding="utf-8")
    for tokenlist in parse_incr(data_file):
        for token_avail in tokenlist:
            #print(token_avail['form'])
            if token_avail['form'] in dico:
                dico[token_avail['form']] += 1
            else:
                dico[token_avail['form']] = 1


    return dico





if __name__ == "__main__":

    path = "/data/dkletz/data/UD/ud-treebanks-v2.11"
    corpus = "UD_French-GSD"
    file = "fr_gsd-ud-dev.conllu"
    dico = get_freq(f"{path}/{corpus}/{file}")
    #print(dico)
    dump(dico, f"dico_{corpus}_{file[:-7]}.joblib")