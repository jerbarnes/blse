import sys
import os
import argparse
from Utils.WordVecs import *
from Utils.Datasets import *
from Utils.utils import*
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC


def scores(model, dataset):
    p = model.predict(dataset._Xtest)
    acc = accuracy_score(dataset._ytest, p)
    prec = per_class_prec(dataset._ytest, p).mean()
    rec = per_class_rec(dataset._ytest, p).mean()
    f1 = macro_f1(dataset._ytest, p)
    return acc, prec, rec, f1


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', default='opener_sents', help="dataset to train and test on (default: opener)")
    args = parser.parse_args()

    vecs = WordVecs('/home/jeremy/NS/Keep/Temp/Exps/EMBEDDINGS/BLSE/google.txt')

    en = General_Dataset(os.path.join('datasets', 'en', args.dataset),
                         vecs, one_hot=False, rep=ave_vecs, lowercase=False)
    en_binary = General_Dataset(os.path.join('datasets', 'en', args.dataset),
                                 vecs, one_hot=False, rep=ave_vecs, binary=True, lowercase=False)

    langs = ['es', 'ca', 'eu']

    for lang in langs:
        print('#### {0} ####'.format(lang))
        cross_dataset = General_Dataset(os.path.join('datasets','trans',lang, args.dataset),
                                        vecs, one_hot=False, rep=ave_vecs, lowercase=False)
        binary_cross_dataset = General_Dataset(os.path.join('datasets','trans',lang, args.dataset),
                                               vecs, one_hot=False, rep=ave_vecs,
                                               binary=True, lowercase=False)

        print('-binary-')
        best_c, best_f1 = get_best_C(en_binary, binary_cross_dataset)
        clf = LinearSVC(C=best_c)
        clf.fit(en_binary._Xtrain, en_binary._ytrain)
        acc, prec, rec, f1 = scores(clf, binary_cross_dataset)
        print_prediction(clf, binary_cross_dataset, os.path.join('predictions', lang, 'mt-svm', '{0}-bi.txt'.format(args.dataset)))
        print('acc:   {0:.3f}'.format(acc))
        print('prec:  {0:.3f}'.format(prec))
        print('rec:   {0:.3f}'.format(rec))
        print('f1:    {0:.3f}'.format(f1))

        print('-fine-')
        best_c, best_f1 = get_best_C(en, cross_dataset)
        clf = LinearSVC(C=best_c)
        clf.fit(en._Xtrain, en._ytrain)
        acc, prec, rec, f1 = scores(clf, cross_dataset)
        print_prediction(clf, cross_dataset, os.path.join('predictions', lang, 'mt-svm', '{0}-4cls.txt'.format(args.dataset)))
        print('acc:   {0:.3f}'.format(acc))
        print('prec:  {0:.3f}'.format(prec))
        print('rec:   {0:.3f}'.format(rec))
        print('f1:    {0:.3f}'.format(f1))


if __name__ == '__main__':
    main()
