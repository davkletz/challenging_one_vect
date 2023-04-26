


path = "/data/dkletz/Other_exp/AvecMatthieu/LSTM_ambiguity/language_model"
file = "training_corpus_word_frequencies.txt"

with open(path + file, "rb") as f:
    for line_available in f.readline():
        print(line_available)