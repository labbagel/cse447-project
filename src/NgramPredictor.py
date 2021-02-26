import nltk, os
import dill
import pickle
from nltk.lm import MLE #https://www.nltk.org/api/nltk.lm.html
from nltk.util import ngrams

class NgramPredictor():
    def __init__(self, n):
        self.n = n
        self.model = MLE(n)
    
    def add_tokens(self, tokens, vocab, file_name):
        print("Now loading", file_name)
        path = "data\\" + file_name
        with open(path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            for line in lines:
            # Convert line (String) into a list
            line_list = list(line)
            tokens.append(line_list)
            for c in line_list:
                # Loop thru the line to add the vocab
                vocab.add(c)
        print("Done!")

    def load_training_data(self, datasets):
        tokens = []
        vocab = set()

        for data in datasets:
            self.add_tokens(tokens, vocab, data)
        return tokens, vocab
    
    def create_ngrams(self, tokens):
        ngrams_list = []
        for i in range(self.n):
            ngrams_list.append(list())
            # index i correponds with i+1 gram
        
        for sentence in tokens:
            for i in range(self.n):
                ngrams_list[i].extend(list(ngrams(sentence, i+1)))
        return ngrams_list
        #unigrams.extend(list(ngrams(sentence, 1)))
        #bigrams.extend(list(ngrams(sentence, 2)))

    def fit(self, ngrams, vocab):
        for gram in ngrams:
            self.model.fit([gram], vocabulary_text=vocab)

    def predict(self, context):
        #https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
        #https://www.programiz.com/python-programming/methods/built-in/sorted
        #https://stackoverflow.com/questions/60295058/nltk-mle-model-clarification-trigrams-and-greater
        # Convert predictions into a dictionary
        predictions = dict(self.model.context_counts(self.model.vocab.lookup(context)))
        # Sort the predictions by num of occurances (dict.values)
        zipped = sorted(predictions.items(), key=lambda item: item[1], reverse=True)
        # Get top three
        top_three = zipped[0:4] # the prediction might contain space
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
        # Change your model name here
        model_path += '\\ngram_model_alan.pkl'
        with open(model_path, 'rb') as fin:
            self.model = dill.load(fin)