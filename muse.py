import sys
import os
import argparse
import numpy as np
from sklearn.svm import LinearSVC
from Utils.Datasets import *
from Utils.WordVecs import *
from Utils.MyMetrics import *
from project_translate import *

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-vec_dir', default='../deployment/MUSE/', help=" directory that hold MUSE vectors")
    parser.add_argument('-dataset', default='opener', help="dataset to train and test on (default: opener)")
    parser.add_argument('-bi', help='List of booleans. True is only binary, False is only 4 class. True False is both. (default: [True, False])',
                        default = [True, False], nargs='+', type=str2bool)
    args = parser.parse_args()
    
    # Loop over the three languages
    for lang in [ 'ca', 'eu', 'es']:
        print('################ {0} ##############'.format(lang))
        
        # Import monolingual vectors
        print('importing word embeddings')
        src_vecs = WordVecs(os.path.join(args.vec_dir, 'en-{0}'.format(lang), 'muse-en.txt'))
        trg_vecs = WordVecs(os.path.join(args.vec_dir, 'en-{0}'.format(lang), 'muse-{0}.txt'.format(lang)))

        # Import datasets (representation will depend on final classifier)
        print('importing datasets')
        binary_dataset = General_Dataset(os.path.join('datasets', 'en', args.dataset), src_vecs,
                                      binary=True, rep=ave_vecs, one_hot=False, lowercase=False)
        binary_cross_dataset = General_Dataset(os.path.join('datasets', lang, args.dataset), trg_vecs,
                                      binary=True, rep=ave_vecs, one_hot=False, lowercase=False)

        fine_dataset = General_Dataset(os.path.join('datasets', 'en', args.dataset), src_vecs,
                                      binary=False, rep=ave_vecs, one_hot=False, lowercase=False)
        fine_cross_dataset = General_Dataset(os.path.join('datasets', lang, args.dataset), trg_vecs,
                                      binary=False, rep=ave_vecs, one_hot=False, lowercase=False)

        # Train linear SVM classifier
        if True in args.bi:
            best_c, best_f1 = get_best_C(binary_dataset, binary_cross_dataset)
            clf = LinearSVC(C=best_c)
            clf.fit(binary_dataset._Xtrain, binary_dataset._ytrain)
            cpred = clf.predict(binary_cross_dataset._Xtest)
            cf1 = macro_f1(binary_cross_dataset._ytest, cpred)
            print_prediction(clf, binary_cross_dataset, os.path.join('predictions', lang, 'muse', '{0}-bi.txt'.format(args.dataset)))
            print('-binary-')
            print('Acc: {0:.3f}'.format(clf.score(binary_cross_dataset._Xtest, binary_cross_dataset._ytest)))
            print('Macro F1: {0:.3f}'.format(cf1))
            print()

        if False in args.bi:
            best_c, best_f1 = get_best_C(fine_dataset, fine_cross_dataset)
            clf = LinearSVC(C=best_c)
            clf.fit(fine_dataset._Xtrain, fine_dataset._ytrain)
            cpred = clf.predict(fine_cross_dataset._Xtest)
            cf1 = macro_f1(fine_cross_dataset._ytest, cpred)
            print_prediction(clf, fine_cross_dataset, os.path.join('predictions', lang, 'muse', '{0}-4cls.txt'.format(args.dataset)))
            print('-fine-')
            print('Acc: {0:.3f}'.format(clf.score(fine_cross_dataset._Xtest, fine_cross_dataset._ytest)))
            print('Macro F1: {0:.3f}'.format(cf1))
            print()


if __name__ == '__main__':
    main()
