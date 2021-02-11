import nltk
import dill as pickle
from nltk.lm.preprocessing import padded_everygram_pipeline

def add_tokens(tokens, file_path):
    file = open(file_path, "r")
    line = file.readline()
    while line != "":
        index = 0
        while line[index] == " ":
            index += 1
        while index < len(line):
            tokens.append(line[index])
            index += 2
        line = file.readline()
    file.close()
    return tokens
        
print("Loading data...")
tokens = []
filepath = ["./datasets/english_dataset.txt", "./datasets/french_dataset.txt", "./datasets/russian_dataset.txt", "./datasets/spanish_dataset.txt"]
# for path in filepath:
#     tokens = add_tokens(tokens, path)
wordcount = 0
    
print("Loading English dataset...")
file = open("./datasets/english_dataset.txt", "r")
line = file.readline()
while wordcount < 1500000:
    index = 0
    sentence = []
    while index < len(line) and line[index] == " ":
        index += 1
    while index < len(line):
        sentence.append(line[index])
        if (index > 0 and line[index] == " " and line[index - 1] != " "):
            wordcount += 1
        index += 2
    if len(sentence) != 0:
        tokens.append(sentence)
    line = file.readline()
file.close()

print("Done")

print("Loading Russian dataset...")

file = open("./datasets/russian_dataset.txt", "r")
line = file.readline()
while wordcount < 460000:
    index = 0
    sentence = []
    while index < len(line) and line[index] == " ":
        index += 1
    while index < len(line):
        sentence.append(line[index])
        if (index > 0 and line[index] == " " and line[index - 1] != " "):
            wordcount += 1
        index += 2
    if len(sentence) != 0:
        tokens.append(sentence)
    line = file.readline()
file.close()

print("Done")

print("Loading French dataset...")

file = open("./datasets/french_dataset.txt", "r")
line = file.readline()
while wordcount < 110000:
    index = 0
    sentence = []
    while index < len(line) and line[index] == " ":
        index += 1
    while index < len(line):
        sentence.append(line[index])
        if (index > 0 and line[index] == " " and line[index - 1] != " "):
            wordcount += 1
        index += 2
    if len(sentence) != 0:
        tokens.append(sentence)
    line = file.readline()
file.close()

print("Done loading all data and creating tokens")
print("Number of tokens:", len(tokens))

# fourgram = nltk.ngrams(tokens, 4)
# freq_four = nltk.FreqDist(fourgram)

n = 2
model = nltk.lm.MLE(n)
print("Generating training data and vocabulary...")
train_data, padded_sents = padded_everygram_pipeline(n, tokens)
print("Done")
# print("train data:", train_data)
# for ngram in train_data:
#     print(ngram)

# print("padded sents:", padded_sents)
# for sent in padded_sents:
#     print(sent)

print("Fitting model...")
model.fit(train_data, padded_sents)
# model.fit(train_data, padd)
print("Done")
print(model.vocab)

print("Saving model to ngram_model.pkl")
with open('ngram_model.pkl', 'wb') as fout:
    pickle.dump(model, fout)

# with open('kilgariff_ngram_model.pkl', 'rb') as fin:
#     model_loaded = pickle.load(fin)


# print ("Most common bigrams: ", freq_four.most_common(5))