import os, re
import numpy as np
from Utils.Representations import *

class ProjectionDataset():
    """
    A wrapper for the translation dictionary. The translation dictionary
    should be word to word translations separated by a tab. The
    projection dataset only includes the translations that are found
    in both the source and target vectors.
    """
    def __init__(self, translation_dictionary, src_vecs, trg_vecs):
        (self._Xtrain, self._Xdev, self._ytrain,
         self._ydev) = self.getdata(translation_dictionary, src_vecs, trg_vecs)

    def train_dev_split(self, x, train=.9):
        # split data into training and development, keeping /train/ amount for training.
        train_idx = int(len(x)*train)
        return x[:train_idx], x[train_idx:]

    def getdata(self, translation_dictionary, src_vecs, trg_vecs):
        x, y = [], []
        with open(translation_dictionary) as f:
            for line in f:
                src, trg = line.split()
                try:
                    _ = src_vecs[src]
                    _ = trg_vecs[trg]
                    x.append(src)
                    y.append(trg)
                except:
                    pass
        xtr, xdev = self.train_dev_split(x)
        ytr, ydev = self.train_dev_split(y)
        return xtr, xdev, ytr, ydev

class General_Dataset(object):
    """This class takes as input the directory of a corpus annotated for 4 levels
    sentiment. This directory should have 4 .txt files: strneg.txt, neg.txt,
    pos.txt and strpos.txt. It also requires a word embedding model, such as
    those used in word2vec or GloVe.

    binary: instead of 4 classes you have binary (pos/neg). Default is False

    one_hot: the y labels are one hot vectors where the correct class is 1 and
             all others are 0. Default is True.

    dtype: the dtype of the np.array for each vector. Default is np.float32.

    rep: this determines how the word vectors are represented.

         ave_vecs: each sentence is represented as the average of all of the
                    word vectors in the sentence.

         idx_vecs: each sentence is respresented as a list of word ids given by
                    the word-2-idx dictionary.

         words: each sentence is a list of the tokens in the sentence
    """

    def __init__(self, DIR, model, binary=False, one_hot=True,
                 dtype=np.float32, rep=ave_vecs, lowercase=True):

        self.rep = rep
        self.one_hot = one_hot
        self.lowercase = lowercase

        Xtrain, Xdev, Xtest, ytrain, ydev, ytest = self.open_data(DIR, model, binary, rep)


        self._Xtrain = Xtrain
        self._ytrain = ytrain
        self._Xdev = Xdev
        self._ydev = ydev
        self._Xtest = Xtest
        self._ytest = ytest
        self._num_examples = len(self._Xtrain)

    def to_array(self, integer, num_labels):
        """quick trick to convert an integer to a one hot vector that
        corresponds to the y labels"""
        integer = integer - 1
        return np.array(np.eye(num_labels)[integer])

    def open_data(self, DIR, model, binary, rep):
        if binary:
            ##################
            # Binary         #
            ##################
            train_neg = getMyData(os.path.join(DIR, 'train/neg.txt'),
                                  0, model, encoding='latin',
                                  representation=rep)
            train_neg2 = getMyData(os.path.join(DIR, 'train/strneg.txt'),
                                  0, model, encoding='latin',
                                  representation=rep)
            train_pos = getMyData(os.path.join(DIR, 'train/pos.txt'),
                                  1, model, encoding='latin',
                                  representation=rep)
            train_pos2 = getMyData(os.path.join(DIR, 'train/strpos.txt'),
                                  1, model, encoding='latin',
                                  representation=rep)
            dev_neg = getMyData(os.path.join(DIR, 'dev/neg.txt'),
                                0, model, encoding='latin',
                                representation=rep)
            dev_neg2 = getMyData(os.path.join(DIR, 'dev/strneg.txt'),
                                0, model, encoding='latin',
                                representation=rep)
            dev_pos = getMyData(os.path.join(DIR, 'dev/pos.txt'),
                                1, model, encoding='latin',
                                representation=rep)
            dev_pos2 = getMyData(os.path.join(DIR, 'dev/strpos.txt'),
                                1, model, encoding='latin',
                                representation=rep)
            test_neg = getMyData(os.path.join(DIR, 'test/neg.txt'),
                                 0, model, encoding='latin',
                                 representation=rep)
            test_neg2 = getMyData(os.path.join(DIR, 'test/strneg.txt'),
                                 0, model, encoding='latin',
                                 representation=rep)
            test_pos = getMyData(os.path.join(DIR, 'test/pos.txt'),
                                 1, model, encoding='latin',
                                 representation=rep)
            test_pos2 = getMyData(os.path.join(DIR, 'test/strpos.txt'),
                                 1, model, encoding='latin',
                                 representation=rep)

            

            traindata = train_pos + train_pos2 + train_neg + train_neg2
            devdata = dev_pos + dev_pos2 + dev_neg + dev_neg2
            testdata = test_pos + test_pos2 + test_neg + test_neg2

            # Set up vocab now
            self.vocab = set()

            
            # Training data
            Xtrain = [data for data, y in traindata]
            if self.lowercase:
                Xtrain = [[w.lower() for w in sent] for sent in Xtrain]
            if self.one_hot is True:
                ytrain = [self.to_array(y, 2) for data, y in traindata]
            else:
                ytrain = [y for data, y in traindata]
            self.vocab.update(set([w for i in Xtrain for w in i]))

            # Dev data
            Xdev = [data for data, y in devdata]
            if self.lowercase:
                Xdev = [[w.lower() for w in sent] for sent in Xdev]
            if self.one_hot is True:
                ydev = [self.to_array(y, 2) for data, y in devdata]
            else:
                ydev = [y for data, y in devdata]
            self.vocab.update(set([w for i in Xdev for w in i]))

            # Test data
            Xtest = [data for data, y in testdata]
            if self.lowercase:
                Xtest = [[w.lower() for w in sent] for sent in Xtest]
            if self.one_hot is True:
                ytest = [self.to_array(y, 2) for data, y in testdata]
            else:
                ytest = [y for data, y in testdata]
            self.vocab.update(set([w for i in Xtest for w in i]))
        
        else:
            ##################
            # 4 CLASS        #
            ##################
            train_strneg = getMyData(os.path.join(DIR, 'train/strneg.txt'),
                                  0, model, encoding='latin',
                                  representation=rep)
            train_strpos = getMyData(os.path.join(DIR, 'train/strpos.txt'),
                                  3, model, encoding='latin',
                                  representation=rep)
            train_neg = getMyData(os.path.join(DIR, 'train/neg.txt'),
                                  1, model, encoding='latin',
                                  representation=rep)
            train_pos = getMyData(os.path.join(DIR, 'train/pos.txt'),
                                  2, model, encoding='latin',
                                  representation=rep)
            dev_strneg = getMyData(os.path.join(DIR, 'dev/strneg.txt'),
                                0, model, encoding='latin',
                                representation=rep)
            dev_strpos = getMyData(os.path.join(DIR, 'dev/strpos.txt'),
                                3, model, encoding='latin',
                                representation=rep)
            dev_neg = getMyData(os.path.join(DIR, 'dev/neg.txt'),
                                1, model, encoding='latin',
                                representation=rep)
            dev_pos = getMyData(os.path.join(DIR, 'dev/pos.txt'),
                                2, model, encoding='latin',
                                representation=rep)
            test_strneg = getMyData(os.path.join(DIR, 'test/strneg.txt'),
                                 0, model, encoding='latin',
                                 representation=rep)
            test_strpos = getMyData(os.path.join(DIR, 'test/strpos.txt'),
                                 3, model, encoding='latin',
                                 representation=rep)
            test_neg = getMyData(os.path.join(DIR, 'test/neg.txt'),
                                 1, model, encoding='latin',
                                 representation=rep)
            test_pos = getMyData(os.path.join(DIR, 'test/pos.txt'),
                                 2, model, encoding='latin',
                                 representation=rep)

            traindata = train_pos + train_neg + train_strneg + train_strpos
            devdata = dev_pos + dev_neg + dev_strneg + dev_strpos
            testdata = test_pos + test_neg + test_strneg + test_strpos

            self.vocab = set()

            # Training data
            Xtrain = [data for data, y in traindata]
            if self.lowercase:
                Xtrain = [[w.lower() for w in sent] for sent in Xtrain]
            if self.one_hot is True:
                ytrain = [self.to_array(y, 4) for data, y in traindata]
            else:
                ytrain = [y for data, y in traindata]
            self.vocab.update(set([w for i in Xtrain for w in i]))

            # Dev data
            Xdev = [data for data, y in devdata]
            if self.lowercase:
                Xdev = [[w.lower() for w in sent] for sent in Xdev]
            if self.one_hot is True:
                ydev = [self.to_array(y, 4) for data, y in devdata]
            else:
                ydev = [y for data, y in devdata]
            self.vocab.update(set([w for i in Xdev for w in i]))

            # Test data
            Xtest = [data for data, y in testdata]
            if self.lowercase:
                Xtest = [[w.lower() for w in sent] for sent in Xtest]
            if self.one_hot is True:
                ytest = [self.to_array(y, 4) for data, y in testdata]
            else:
                ytest = [y for data, y in testdata]
            self.vocab.update(set([w for i in Xtest for w in i]))

        if self.rep is not words:
            Xtrain = np.array(Xtrain)
            Xdev = np.array(Xdev)
            Xtest = np.array(Xtest)
        ytrain = np.array(ytrain)
        ydev = np.array(ydev)
        ytest = np.array(ytest)

        return Xtrain, Xdev, Xtest, ytrain, ydev, ytest

