import argparse
from blse import BLSE

from Utils.utils import str2bool, get_best_run
from Utils.Datasets import General_Dataset
from Utils.WordVecs import WordVecs
from Utils.Representations import words

import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-da', '--dataset',
                        help="dataset to predict on")
    parser.add_argument('-sd', '--savedir',
                        help="directory where trained BLSE weights are found (default: ./models/blse)",
                        default='models/blse')
    parser.add_argument('-bi', '--binary', default=True, type=str2bool)
    parser.add_argument('-sv', '--src_vecs',
                        help=" source language vectors (default: GoogleNewsVecs )",
                        default='google.txt')
    parser.add_argument('-tv', '--trg_vecs',
                        help=" target language vectors (default: SGNS on Wikipedia)",
                        default='sg-300-es.txt')
    parser.add_argument('-tl', '--target_lang',
                        help="target language: es, ca, eu, en (default: es)",
                        default='es')
    parser.add_argument('-o', '--outfile', help="name of prediction file",
                        default="predictions.txt")
    args = parser.parse_args()

    # The data for inference should be a plaintext file with a single tokenized sentence per line
    inference_data = [line.split() for line in open(args.dataset)]

    # Import monolingual vectors
    print('importing word embeddings')
    src_vecs = WordVecs(args.src_vecs)
    trg_vecs = WordVecs(args.trg_vecs)

    if args.binary:
        output_dim = 2
        b = 'bi'
    else:
        output_dim = 4
        b = '4cls'

    # Set up model
    blse = BLSE(src_vecs, trg_vecs, None, inference_data, None,
                projection_loss="mse",
                output_dim=output_dim,
                src_syn1=None, src_syn2=None, src_neg=None)


    # Get best dev f1 and weights
    best_f1, best_params, best_weights = get_best_run(args.savedir)
    blse.load_weights(best_weights)
    print()
    print('Dev set')
    print('best dev f1: {0:.3f}'.format(best_f1))
    print('parameters: epochs {0} batch size {1} alpha {2}'.format(*best_params))

    labels = blse.predict_labels(inference_data, src=False)

    with open(args.outfile, "w") as outfile:
        for l in labels:
            outfile.write(l + "\n")

if __name__ == '__main__':
    main()
