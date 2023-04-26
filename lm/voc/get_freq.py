
from joblib import dump

path = "/data/dkletz/Other_exp/AvecMatthieu/LSTM_ambiguity/language_model/"
file = "training_corpus_word_frequencies.txt"


results = {}
with open(path + file, "rb") as f:

    for line_available in f.readlines():
        print(line_available)

        content = line_available.split()
        print(content)

        if len(content) == 2:

            word = str(content[0])
            if word[-1] == ',':
                word = word[:-1]
            freq = int(content[1])
            results[word] = freq



dump(results, "word_freq.joblib")



