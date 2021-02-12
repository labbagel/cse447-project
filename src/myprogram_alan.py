#!/usr/bin/env python
import os
import string
import random
import dill as pickle
from NgramPredictor import NgramPredictor
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


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
        tokens, vocab = model.load_training_data(["english_dataset.txt", "russian_dataset.txt",
                                                "french_dataset.txt", "spanish_dataset.txt"])
        ngrams = model.create_ngrams(tokens)
        print('Training')
        model.fit(ngrams, vocab)
        print('Saving model')
        model.save_model(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = NgramPredictor(n=2)
        model.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        
        while(True):
            text = input("Enter your line: ")
            
            if text == "stop the script":
                print("Ending The Program.....")
                break
            else:
                try:
                    pred = model.predict(text[-1])
                    print(pred)
                except:
                    continue

    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))

if __name__ == '__main__':
    main()