import nltk
import dill as pickle
from nltk.lm import MLE #https://www.nltk.org/api/nltk.lm.html
from nltk.lm.preprocessing import padded_everygram_pipeline

def add_tokens(tokens, vocab, file_name):
  print("Now loading", file_name)
  path = "/content/drive/MyDrive/CSE447/Project/Data_cleaned/" + file_name
  with open(path, "r") as file:
    lines = file.readlines()
    for line in lines:
      index = 0
      sentence = []
      # start from the index of a character
      while index < len(line) and line[index] == " ":
          index += 1
      # start storing sentence
      while index < len(line):
        vocab.add(line[index])
        sentence.append(line[index])
        if (index > 0 and line[index] == " " and line[index - 1] != " "):
            wordcount += 1
        index += 2
      if len(sentence) != 0:
          tokens.append(sentence)
  print("Done!")

tokens = []
vocab = set()
# Add all dataset
datasets = ["english_dataset.txt", "russian_dataset.txt", "french_dataset.txt", "spanish_dataset.txt"]
for data in datasets:
  add_tokens(tokens, vocab, data)

print("Number of tokens:", len(tokens))

## Create ngrams
unigrams = []
bigrams = []
for sentence in tokens:
  unigrams.extend(list(ngrams(sentence, 1)))
  bigrams.extend(list(ngrams(sentence, 2)))

n = 2
model = MLE(n)
print("Fitting bigrams...")
model.fit([bigrams], vocabulary_text=vocab)
print("Fitting unigrams...")
model.fit([unigrams])
print("Done")
# Predictions
def predict(context):
    #https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
    #https://www.programiz.com/python-programming/methods/built-in/sorted
    #https://stackoverflow.com/questions/60295058/nltk-mle-model-clarification-trigrams-and-greater
    predictions = dict(lm.context_counts(lm.vocab.lookup(context)))
    zipped = sorted(predictions.items(), key=lambda item: item[1], reverse=True)
    top_three = zipped[0:4] # the first prediction is always space
    chars = [top_three[0][0], top_three[1][0], top_three[2][0], top_three[3][0]]
    if ' ' in chars:
        chars.remove(' ')
        return chars
    else:
        return chars[:-1]


print("Saving model to ngram_model2.pkl")
with open('ngram_model.pkl', 'wb') as fout:
    pickle.dump(model, fout)