from joblib import dump

path = "/data/dkletz/Other_exp/AvecMatthieu/LSTM_ambiguity/language_model/"
file = "training_corpus_vocab.txt"

results = {}
idx_to_words = {}
i = 0
with open(path + file, "rb") as f:
    for line_available in f.readlines():
        # print(line_available)

        content = line_available.split()
        # print(content)

        if len(content) == 1:

            word = str(content[0])
            while word[-1] in [",", "'", '"']:
                word = word[:-1]

            while word[:2] in ["b'", 'b"']:
                word = word[2:]

            results[word] = i
            idx_to_words[i] = word

        else:
            print("pb")
            print(content)

        i+=1

dump(results, "word_to_idx.joblib")

dump(idx_to_words, "idx_to_word.joblib")



