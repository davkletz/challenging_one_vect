from joblib import dump
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize import sent_tokenize
from random import random
from transformers import BertTokenizer, RobertaTokenizer
from joblib import dump

detokenizer = TreebankWordDetokenizer()

DICO_EQUIV = {"magazine": 'mag', "acad": 'acad', "fiction": "fic", "newspaper": 'news', "spoken": 'spok'}

list_mod_aux =  ["are", "can", "could", "dare", "did", "does", "do", "has",  "have", "is", "may", "might", "must", "need", "ought", "shall", "should", "was", "were", "will", "would", "am"]






def get_row(line):
    row = []
    current_w = ''
    for element in line:
        if element != '\t':
            if element != '\n':
                current_w += element

        else:
            row.append(current_w)
            current_w = ''



    row.append(current_w)

    return row



def catch_sentences(path_data, tokenizer, detail_toks, nb_sents):

    tsv_file = open(path_data, encoding='unicode_escape')

    lines = tsv_file.readlines()

    tokens = []
    POS = []

    nb_toks = 0

    for line_available in lines:
        nb_toks+=1

        '''if nb_toks >1000:
            break'''
        row = get_row(line_available)
        current_word = row[0]
        if current_word not in ["", "//", "<p>"] and current_word[:2] != "##":

            tokens.append(current_word)
            POS.append(row[-1])

        elif  current_word == "<p>":
            tokens.append(".")
            POS.append(row[-1])

    sentence_detokenized = detokenizer.detokenize(tokens)


    tokens_sentence = []
    for sentence in sent_tokenize(sentence_detokenized):
        nb_sents+=1
        tokens_sentence.append(tokenizer.tokenize(sentence))





    for sentence_avail in tokens_sentence:

        for token_available in sentence_avail:
            if token_available not in detail_toks:
                detail_toks[token_available] = 0
            detail_toks[token_available] += 1

    return detail_toks, nb_sents







def match_sentences_POS(all_tokens, detail_sentences, POS):
    list_sentences = []#all
    list_POS = []#all




    indice_car_in_token = 0

    indice_token_in_all = 0




    for indice_sentence, sentence_available in enumerate(detail_sentences):
        tokens_of_current_sentence = []
        POS_of_current_sentence = []

        for indice_car_in_sentence in range(len(sentence_available)):
            #print('\nH\n')

            current_car_in_sentence = sentence_available[indice_car_in_sentence]
            #print(current_car_in_sentence)
            #print(indice_car_in_token)
            #print(all_tokens[indice_token_in_all])

            current_car_in_tokens = all_tokens[indice_token_in_all][indice_car_in_token]
            #print(current_car_in_tokens)


            if current_car_in_tokens == current_car_in_sentence:
                indice_car_in_token += 1

            if indice_car_in_token >= len(all_tokens[indice_token_in_all]):
                tokens_of_current_sentence.append(all_tokens[indice_token_in_all])
                POS_of_current_sentence.append(POS[indice_token_in_all])
                indice_token_in_all += 1
                indice_car_in_token = 0

        list_sentences.append(tokens_of_current_sentence)
        list_POS.append(POS_of_current_sentence)


    return list_sentences, list_POS











if __name__ == '__main__':

    #tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')



    path = "/home/dkletz/COCA/"

    NPIs_to_catch = ["some", "somebody", "someone", "something", "sometime", "somewhere"]

    list_corpuses = ["magazine", "acad", "fiction", "newspaper"]


    detail_pos = {}
    detail_toks = {}
    all_pos = []
    all_toks = []

    nb_sents = 0

    nb_files = 0

    for corpus_available in list_corpuses:
        detail_nb_with_corpus = 0
        detail_nb_without_corpus = 0

        print(f"current : {corpus_available}")

        list_files = [f'{path}{corpus_available}/wlp_{DICO_EQUIV[corpus_available]}_{y}.txt' for y in range(1990, 2013)]
        for file_available in list_files:
            nb_files+=1
            detail_toks, nb_sents = catch_sentences(file_available, tokenizer, detail_toks, nb_sents)


    print(nb_sents)
    print(nb_files)
    dump(detail_toks, f"Rob_detail_toks.joblib")




