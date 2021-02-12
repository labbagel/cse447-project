import nltk, os
import dill
import pickle
from nltk.lm import MLE #https://www.nltk.org/api/nltk.lm.html
from nltk.lm.preprocessing import padded_everygram_pipeline

class NgramPredictor():
    def __init__(self, n):
        self.n = n
        self.model = MLE(n)
    
    def add_tokens(self, tokens, vocab, file_name):
        print("Now loading", file_name)
        print(os.getcwd())
        path = "data\\" + file_name
        print(path)
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

    def load_training_data(self, datasets):
        tokens = []
        vocab = set()

        for data in datasets:
            self.add_tokens(tokens, vocab, data)
        return tokens, vocab
    
    def create_ngrams(self, tokens):
        ngrams = []
        for i in range(self.n):
            ngrams.append(list())
            # index i correponds with i+1 gram
        
        for sentence in tokens:
            for i in range(self.n):
                ngrams[i].extend(list(ngrams(sentence, i+1)))
        return ngrams
        #unigrams.extend(list(ngrams(sentence, 1)))
        #bigrams.extend(list(ngrams(sentence, 2)))

    def fit(self, ngrams, vocab):
        for gram in ngrams:
            self.model.fit([gram], vocabulary_text=vocab)

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
    
    def save_model(self, work_dir):
        with open('ngram_model_alan.pkl', 'wb') as fout:
            dill.dump(self.model, fout)
    
    @classmethod
    def save_variables(variables, name):
        with open(name+'.pkl', 'wb') as fout:
            pickle.dump(variables, f)
    
    def load_model(self, model_path):
        with open(model_path, 'rb') as fin:
            self.model = dill.load(fin)