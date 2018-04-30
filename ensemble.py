import sys
import os
import argparse
import numpy as np
from blse import *
from mt import *
from artetxe import *
from sklearn.ensemble import RandomForestClassifier
from Utils.utils import *


def get_randomforest_parameters(X_train, y_train,
                                X_dev, y_dev):
    best_f1 = 0
    best_n = 0
    best_f = 0
    n_estimators = np.arange(10, 100, 30)
    for n in n_estimators:
        clf = RandomForestClassifier(n_estimators=n)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_dev)
        f1 = per_class_f1(y_dev, pred).mean()
        if f1 > best_f1:
            best_f1 = f1
            best_n = n
    return best_f1, best_n

def main():
    parser = argparse.ArgumentParser(description='Ensemble approach')
    parser.add_argument('-l', '--lang', default='es', help='language: es, eu, ca')
    parser.add_argument('-d', '--dataset', default='opener_sents', help='dataset to test')
    #parser.add_argument('-m', '--models', default=['mt-svm', 'blse', 'barista-svm', 'artetxe-svm'], nargs='+', type=str)
    parser.add_argument('-m', '--models', default=['mt-svm', 'blse'], nargs='+', type=str)
    parser.add_argument('-bi', default=True, type=str2bool)
    args = parser.parse_args()

    print('Language: {0}'.format(args.lang))
    print('Dataset:  {0}'.format(args.dataset))
    print('Models:   {0}'.format(' '.join(args.models)))
    print('Binary:   {0}'.format(args.bi))
    print()

    # Get first datasets for vocab and BLSE
    blse_dataset = General_Dataset(os.path.join('datasets','en', args.dataset), None,
                              binary=args.bi, one_hot=False,
                              lowercase=True, rep=words)
    
    blse_crossdataset = General_Dataset(os.path.join('datasets', args.lang, args.dataset), None,
                              binary=args.bi, one_hot=False,
                              lowercase=True, rep=words)


    # Load Word Embeddings
    print('Loading embeddings and datasets...')
    src_vecs = WordVecs('/home/jeremy/NS/Keep/Temp/Exps/EMBEDDINGS/BLSE/google.txt')
    trg_vecs = WordVecs('/home/jeremy/NS/Keep/Temp/Exps/EMBEDDINGS/BLSE/sg-300-{0}.txt'.format(args.lang))

    if 'barista-svm' in args.models:
        barista_vecs = WordVecs('/home/jeremy/NS/Keep/Temp/Exps/BWEs_for_CLSA/embeddings/barista/sg-300-window4-negative20_en_{0}.txt'. format(args.lang),
                                vocab=list(blse_dataset.vocab)+list(blse_crossdataset.vocab))

    mt_dataset = General_Dataset(os.path.join('datasets','en', args.dataset), src_vecs,
                              binary=args.bi, one_hot=False,
                              lowercase=False, rep=ave_vecs)

    mt_crossdataset = General_Dataset(os.path.join('datasets', 'trans', args.lang, args.dataset), src_vecs,
                              binary=args.bi, one_hot=False,
                              lowercase=False, rep=ave_vecs)

    if 'barista-svm' in args.models:
        ba_dataset = General_Dataset(os.path.join('datasets', 'en', args.dataset), barista_vecs,
                              binary=args.bi, one_hot=False,
                              lowercase=False, rep=ave_vecs)

    
        ba_crossdataset = General_Dataset(os.path.join('datasets', args.lang, args.dataset), barista_vecs,
                              binary=args.bi, one_hot=False,
                              lowercase=False, rep=ave_vecs)
    

    # set up and train classifiers
    print('Setting up and training individual classifiers...')
    classifiers = {}

    
    # BLSE
    if 'blse' in args.models:
        print('    BLSE...')
        if args.dataset == 'opener':
            if args.bi:
                best_f1, best_params, best_weights = get_best_run('models/aspect_level/all-{0}-bi'.format(args.lang))
            else:
                best_f1, best_params, best_weights = get_best_run('models/aspect_level/all-{0}-4cls'.format(args.lang))
        elif args.dataset == 'opener_sents':
            if args.bi:
                best_f1, best_params, best_weights = get_best_run('models/sent_level/opener-{0}-bi'.format(args.lang))
            else:
                best_f1, best_params, best_weights = get_best_run('models/sent_level/opener-{0}-4cls'.format(args.lang))
        elif args.dataset == 'opener_docs':
            if args.bi:
                best_f1, best_params, best_weights = get_best_run('models/document_level/opener-{0}-bi'.format(args.lang))
            else:
                print('No weights!!!')
        if args.bi:
            blse = BLSE_test(src_vecs, trg_vecs, output_dim=2)
        else:
            blse = BLSE_test(src_vecs, trg_vecs, output_dim=4)
        blse.load_weights(best_weights)
        classifiers['blse'] = {}
        classifiers['blse']['model'] = blse
        classifiers['blse']['crossdataset'] = blse_crossdataset
    
    # MT
    if 'mt-svm' in args.models:
        print('    MT-SVM...')
        best_c, best_f1 = get_best_C(mt_dataset, mt_crossdataset)
        mt_clf = LinearSVC(C=best_c)
        mt_clf.fit(mt_dataset._Xtrain, mt_dataset._ytrain)
        classifiers['mt-svm'] = {}
        classifiers['mt-svm']['model'] = mt_clf
        classifiers['mt-svm']['crossdataset'] = mt_crossdataset

    # Barista
    if 'barista-svm' in args.models:
        print('    BARISTA-SVM...')
        best_c, best_f1 = get_best_C(ba_dataset, ba_crossdataset)
        ba_clf = LinearSVC(C=best_c)
        ba_clf.fit(ba_dataset._Xtrain, ba_dataset._ytrain)
        classifiers['barista-svm'] = {}
        classifiers['barista-svm']['model'] = ba_clf
        classifiers['barista-svm']['crossdataset'] = ba_crossdataset

    # Artetxe
    if 'artetxe-svm' in args.models:
        print('    ARTETXE-SVM...')
        pdataset = ProjectionDataset('lexicons/bingliu_en_{0}.one-2-one.txt'.format(args.lang),
                                 src_vecs, trg_vecs)

        # learn the translation matrix W
        W = get_W(pdataset, src_vecs, trg_vecs)

        # project the source matrix to the new shared space
        src_vecs._matrix = np.dot(src_vecs._matrix, W)

        # get Artetxe datasets
        artetxe_dataset = General_Dataset(os.path.join('datasets', 'en', args.dataset), src_vecs,
                                  binary=args.bi, one_hot=False,
                                  lowercase=False, rep=ave_vecs)

        artetxe_crossdataset = General_Dataset(os.path.join('datasets', args.lang, args.dataset), trg_vecs,
                                  binary=args.bi, one_hot=False,
                                  lowercase=False, rep=ave_vecs)
            
        best_c, best_f1 = get_best_C(artetxe_dataset, artetxe_crossdataset)
        artetxe_clf = LinearSVC(C=best_c)
        artetxe_clf.fit(artetxe_dataset._Xtrain, artetxe_dataset._ytrain)
        classifiers['artetxe-svm'] = {}
        classifiers['artetxe-svm']['model'] = artetxe_clf
        classifiers['artetxe-svm']['crossdataset'] = artetxe_crossdataset


    # Predict training and dev for all models
    print('Collecting predictions from all models...')
    meta_X_preds = []
    for name in classifiers.keys():
        if name == 'blse':
            pred = classifiers[name]['model'].predict(classifiers[name]['crossdataset']._Xtrain).data.numpy()
            meta_X_preds.append(pred)
        else:
            pred = classifiers[name]['model'].decision_function(classifiers[name]['crossdataset']._Xtrain)
            if args.bi:
                pos = pred
                neg = 1 - pred
                pred = np.array(list(zip(neg, pos)))
            meta_X_preds.append(pred)
            
    meta_train_X = np.stack((meta_X_preds))
    shape = meta_train_X.shape
    meta_train_X = meta_train_X.reshape((shape[1], shape[0]*shape[2]))
    meta_train_y = blse_crossdataset._ytrain

    # Predict dev for all models
    meta_X_preds = []
    for name in classifiers.keys():
        if name == 'blse':
            pred = classifiers[name]['model'].predict(classifiers[name]['crossdataset']._Xdev).data.numpy()
            meta_X_preds.append(pred)
        else:
            pred = classifiers[name]['model'].decision_function(classifiers[name]['crossdataset']._Xdev)
            if args.bi:
                pos = pred
                neg = 1 - pred
                pred = np.array(list(zip(neg, pos)))
            meta_X_preds.append(pred)
            
    meta_dev_X = np.stack((meta_X_preds))
    shape = meta_dev_X.shape
    meta_dev_X = meta_dev_X.reshape((shape[1], shape[0]*shape[2]))
    meta_dev_y = blse_crossdataset._ydev

    # Predict test for all models
    meta_X_preds = []
    for name in classifiers.keys():
        if name == 'blse':
            pred = classifiers[name]['model'].predict(classifiers[name]['crossdataset']._Xtest).data.numpy()
            meta_X_preds.append(pred)
        else:
            pred = classifiers[name]['model'].decision_function(classifiers[name]['crossdataset']._Xtest)
            if args.bi:
                pos = pred
                neg = 1 - pred
                pred = np.array(list(zip(neg, pos)))
            meta_X_preds.append(pred)

    meta_test_X = np.stack((meta_X_preds))
    shape = meta_test_X.shape
    meta_test_X = meta_test_X.reshape((shape[1], shape[0]*shape[2]))
    meta_test_y = blse_crossdataset._ytest

    # get parameters for Random Forest Classifier on cross dev
    print('Cross validation...')
    best_f1, best_n = get_randomforest_parameters(meta_train_X, meta_train_y,
                                                          meta_dev_X, meta_dev_y)
    print('Dev f1: {0:.3f}'.format(best_f1))
    print('n_estimators: {0}'.format(best_n))

    # train ensemble classifier
    print('Training ensemble classifier...')
    meta_clf = RandomForestClassifier(n_estimators=best_n)
    meta_clf.fit(meta_train_X, meta_train_y)
    pred = meta_clf.predict(meta_test_X)
    precs = per_class_prec(meta_test_y, pred)
    recs = per_class_rec(meta_test_y, pred)
    f1s = per_class_f1(meta_test_y, pred)
    macro_f1 = per_class_f1(meta_test_y, pred).mean()
    
    # print prediction
    if args.bi:
        b = 'bi'
    else:
        b = '4cls'
    outfile = os.path.join('predictions',  args.lang, 'meta', '{0}-{1}-{2}.txt'.format(b, args.dataset, '-'.join(args.models)))
    print('Printing predictions to {0}...'.format(outfile))
    with open(outfile, 'w') as out:
        for line in pred:
            out.write('{0}\n'.format(line))

    # print results to stdout
    if args.bi:
        print('Results:')
        sys.stdout.write('Neg  {0:.3f}\nPos     {1:.3f}\n'.format(*f1s.reshape(2)))
        sys.stdout.write('\n')
        sys.stdout.write('Macro Precision: {0:.3f}\n'.format(precs.mean()))
        sys.stdout.write('Macro Recall:    {0:.3f}\n'.format(recs.mean()))
        sys.stdout.write('Macro F1:        {0:.3f}\n'.format(macro_f1))
    else:
        print('Results:')
        sys.stdout.write('StrNeg  {0:.3f}\nNeg     {1:.3f}\nPos:    {2:.3f}\nStrPos: {3:.3f}\n'.format(*f1s.reshape(4)))
        sys.stdout.write('\n')
        sys.stdout.write('Macro Precision: {0:.3f}\n'.format(precs.mean()))
        sys.stdout.write('Macro Recall:    {0:.3f}\n'.format(recs.mean()))
        sys.stdout.write('Macro F1:        {0:.3f}\n'.format(macro_f1))


if __name__ == '__main__':
    main()
