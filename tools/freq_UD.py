from io import open
from conllu import parse_incr
from joblib import dump
import sys

def get_freq(path):

    dico = {}

    data_file = open(path,"r", encoding="utf-8")
    for tokenlist in parse_incr(data_file):
        for token_avail in tokenlist:

            z = token_avail['id']
            try:
                z = int(z)
            except:
                print(z)

            if token_avail['form'] in dico:
                dico[token_avail['form']] += 1
            else:
                dico[token_avail['form']] = 1


    return dico





if __name__ == "__main__":

    path = "/data/dkletz/data/UD/ud-treebanks-v2.11"
    #corpus = "UD_French-GSD"
    #file = "fr_gsd-ud-train.conllu"

    #corpus = "UD_Hebrew-HTB"
    #file = "he_htb-ud-train.conllu"

    corpus = sys.argv[1]
    file = sys.argv[2]
    dico = get_freq(f"{path}/{corpus}/{file}")
    #print(dico)
    #dump(dico, f"dico_{corpus}_{file[:-7]}.joblib")

    print(len(dico))
    mm = 0
    for k in dico:

        if dico[k] ==1:
            mm +=1

    print(mm)


    a = list(dico.keys())
    a.sort()

    print(a[:100])