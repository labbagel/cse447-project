#!/usr/bin/env python
import os
import string
import random
import dill as pickle
from NgramPredictor import NgramPredictor
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def load_precomputed_data():
    path = "data\\" + "tokens_and_vocab.pkl"
    with open(path, 'rb') as f:  # Python 3: open(..., 'rb')
        tokens, vocab = pickle.load(f)
    return tokens, vocab


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instantiating model')
        model = NgramPredictor(n=2)
        print('Loading training data')
        # Option 1: Compute again:
        #tokens, vocab = model.load_training_data(["english_dataset.txt", "russian_dataset.txt",
        #                                        "french_dataset.txt", "spanish_dataset.txt"])
        # Option 2: Load precomputed
        tokens, vocab = load_precomputed_data()
        print('Creating_ngrams')
        ngrams = model.create_ngrams(tokens)
        print('Training')
        model.fit(ngrams, vocab)
        print('Saving model')
        model.save_model(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = NgramPredictor(n=2)
        model.load_model(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        
        # Source: https://github.com/Bharath-K3/Next-Word-Prediction-with-NLP-and-Deep-Learning/blob/master/Predictions-1.ipynb
        while(True):
            text = input("Enter your line: ")
            
            if text == "stop the script":
                print("Ending The Program.....")
                break
            else:
                try:
                    pred = model.predict(list(text[-1]))
                    print(pred)
                except:
                    continue

    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))

if __name__ == '__main__':
    main()