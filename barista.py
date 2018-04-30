import sys
import os
import argparse
from Utils.WordVecs import *
from Utils.Datasets import *
from Utils.utils import *
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

def scores(model, dataset, average='macro'):
    p = model.predict(dataset._Xtest)
    acc = accuracy_score(dataset._ytest, p)
    f1 = macro_f1(dataset._ytest, p)
    return acc, f1


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', default='opener', help="dataset to train and test on (default: opener)")
    parser.add_argument('-bi', help='List of booleans. True is only binary, False is only 4 class. True False is both. (default: [True, False])',
                        default = [True, False], nargs='+', type=str2bool)
    args = parser.parse_args()

    langs = ['es', 'ca', 'eu']

    for lang in langs:
        print('#### {0} ####'.format(lang))
        en = General_Dataset(os.path.join('datasets', 'en', args.dataset),
                                        None, one_hot=False, rep=words)
        cross_dataset = General_Dataset(os.path.join('datasets', lang, args.dataset),
                                               None, one_hot=False, rep=words)
        vocab = en.vocab.update(cross_dataset.vocab)
        
        vecs = WordVecs('embeddings/barista/sg-300-window4-negative20_en_{0}.txt'.format(lang),
                        vocab=vocab)
        
        en = General_Dataset(os.path.join('datasets', 'en', args.dataset),
                              vecs, one_hot=False, rep=ave_vecs, lowercase=False)
        en_binary = General_Dataset(os.path.join('datasets', 'en', args.dataset),
                              vecs, one_hot=False, rep=ave_vecs, binary=True, lowercase=False)

        
        cross_dataset = General_Dataset(os.path.join('datasets', lang, args.dataset),
                                               vecs, one_hot=False, rep=ave_vecs, lowercase=False)
        binary_cross_dataset = General_Dataset(os.path.join('datasets', lang, args.dataset),
                                               vecs, one_hot=False, rep=ave_vecs,
                                               binary=True, lowercase=False)
        
        if True in args.bi:
            print('-binary-')
            best_c, best_f1 = get_best_C(en_binary, binary_cross_dataset)
            clf = LinearSVC(C=best_c)
            clf.fit(en_binary._Xtrain, en_binary._ytrain)
            acc, f1 = scores(clf, binary_cross_dataset, 'binary')
            print_prediction(clf, binary_cross_dataset, os.path.join('predictions', lang, 'barista', '{0}-bi.txt'.format(args.dataset)))
            print('acc: {0:.3f}'.format(acc))
            print('f1:  {0:.3f}'.format(f1))

        if False in args.bi:
            print('-fine-')
            best_c, best_f1 = get_best_C(en, cross_dataset)
            clf = LinearSVC(C=best_c)
            clf.fit(en._Xtrain, en._ytrain)
            acc, f1 = scores(clf, cross_dataset)
            print_prediction(clf, cross_dataset, os.path.join('predictions', lang, 'barista', '{0}-4cls.txt'.format(args.dataset)))

            print('acc: {0:.3f}'.format(acc))
            print('f1:  {0:.3f}'.format(f1))


if __name__ == '__main__':
    main()
    
