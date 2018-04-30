import sys
import os
import argparse
from Utils.WordVecs import *
from Utils.Datasets import *
from Utils.MyMetrics import *
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

def scores(model, dataset, average='macro'):
    p = model.predict(dataset._Xtest)
    acc = accuracy_score(dataset._ytest, p)
    f1 = macro_f1(dataset._ytest, p)
    return acc, f1

def to_array(X,n=2):
    return np.array([np.eye(n)[x] for x in X])

def macro_f1(y, pred):
    """Get the per class f1 score"""
    
    num_classes = len(set(y))
    y = to_array(y, num_classes)
    pred = to_array(pred, num_classes)
    
    results = []
    for j in range(num_classes):
        class_y = y[:,j]
        class_pred = pred[:,j]
        mm = MyMetrics(class_y, class_pred, one_hot=False, average='binary')
        prec, rec, f1 = mm.get_scores()
        results.append([f1])
    return np.array(results).mean()

def get_best_C(dataset, cross_dataset):
    """
    Find the best parameters on the dev set.
    """
    best_f1 = 0
    best_c = 0

    labels = sorted(set(dataset._ytrain))

    test_cs = [0.001, 0.003, 0.006, 0.009,
                   0.01,  0.03,  0.06,  0.09,
                   0.1,   0.3,   0.6,   0.9,
                   1,       3,    6,     9,
                   10,      30,   60,    90]
    for i, c in enumerate(test_cs):

        sys.stdout.write('\rRunning cross-validation: {0} of {1}'.format(i+1, len(test_cs)))
        sys.stdout.flush()

        clf = LinearSVC(C=c)
        h = clf.fit(dataset._Xtrain, dataset._ytrain)
        pred = clf.predict(cross_dataset._Xdev)
        dev_f1 = macro_f1(cross_dataset._ydev, pred)
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_c = c

    print()
    print('Best F1 on dev data: {0:.3f}'.format(best_f1))
    print('Best C on dev data: {0}'.format(best_c))

    return best_c, best_f1

def print_prediction(model, cross_dataset, outfile):
    prediction = model.predict(cross_dataset._Xtest)
    with open(outfile, 'w') as out:
        for line in prediction:
            out.write('{0}\n'.format(line))

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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
        
        vecs = WordVecs('/home/jeremy/NS/Keep/Temp/Exps/BWEs_for_CLSA/embeddings/barista/sg-300-window4-negative20_en_{0}.txt'.format(lang),
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
            print_prediction(clf, binary_cross_dataset, os.path.join('predictions', lang, 'barista-svm', '{0}-bi.txt'.format(args.dataset)))
            print('acc: {0:.3f}'.format(acc))
            print('f1:  {0:.3f}'.format(f1))

        if False in args.bi:
            print('-fine-')
            best_c, best_f1 = get_best_C(en, cross_dataset)
            clf = LinearSVC(C=best_c)
            clf.fit(en._Xtrain, en._ytrain)
            acc, f1 = scores(clf, cross_dataset)
            print_prediction(clf, cross_dataset, os.path.join('predictions', lang, 'barista-svm', '{0}-4cls.txt'.format(args.dataset)))

            print('acc: {0:.3f}'.format(acc))
            print('f1:  {0:.3f}'.format(f1))


if __name__ == '__main__':
    main()
    
