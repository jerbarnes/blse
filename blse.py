import sys
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam


from Utils.Datasets import *
from Utils.WordVecs import *
from Utils.utils import *

from sklearn.metrics import f1_score, accuracy_score, precision_score, \
                            recall_score, confusion_matrix
from scipy.spatial.distance import cosine


class BLSE(nn.Module):
    """
    Bilingual Sentiment Embeddings

    Parameters:

        src_vecs: WordVecs instance with the embeddings from the source language
        trg_vecs: WordVecs instance with the embeddings from the target language
        pdataset: Projection_Dataset from source to target language
        cdataset: Source sentiment dataset
        projection_loss: the distance metric to use for the projection loss
                         can be either mse (default) or cosine
        output_dim: the number of class labels to predict (default: 4)

    Optional:

        src_syn1: a list of positive sentiment words in the source language
        src_syn2: a second list of positive sentiment words in the
                  source language. This must be of the same length as
                  src_syn1 and should not have overlapping vocabulary
        src_neg : a list of negative sentiment words in the source language
                  this must be of the same length as src_syn1
        trg_syn1: a list of positive sentiment words in the target language
        trg_syn2: a second list of positive sentiment words in the
                  target language. This must be of the same length as
                  trg_syn1 and should not have overlapping vocabulary
        trg_neg : a list of negative sentiment words in the target language
                  this must be of the same length as trg_syn1



    """

    def __init__(self, src_vecs, trg_vecs, pdataset,
                 cdataset, trg_dataset,
                 projection_loss='mse',
                 output_dim=4,
                 src_syn1=None, src_syn2=None, src_neg=None,
                 trg_syn1=None, trg_syn2=None, trg_neg=None,
                 ):
        super(BLSE, self).__init__()

        # Embedding matrices
        self.semb = nn.Embedding(src_vecs.vocab_length, src_vecs.vector_size)
        self.semb.weight.data.copy_(torch.from_numpy(src_vecs._matrix))
        self.sw2idx = src_vecs._w2idx
        self.sidx2w = src_vecs._idx2w
        self.temb = nn.Embedding(trg_vecs.vocab_length, trg_vecs.vector_size)
        self.temb.weight.data.copy_(torch.from_numpy(trg_vecs._matrix))
        self.tw2idx = trg_vecs._w2idx
        self.tidx2w = trg_vecs._idx2w

        # Projection vectors
        self.m = nn.Linear(src_vecs.vector_size,
                           src_vecs.vector_size,
                           bias=False)
        self.mp = nn.Linear(trg_vecs.vector_size,
                            trg_vecs.vector_size,
                            bias=False)

        # Classifier
        self.clf = nn.Linear(src_vecs.vector_size, output_dim)

        # Loss Functions
        self.criterion = nn.CrossEntropyLoss()
        if projection_loss == 'mse':
            self.proj_criterion = mse_loss
        elif projection_loss == 'cosine':
            self.proj_criterion = cosine_loss

        # Optimizer
        self.optim = torch.optim.Adam(self.parameters())

        # Datasets
        self.pdataset = pdataset
        self.cdataset = cdataset
        self.trg_dataset = trg_dataset
        self.src_syn1 = src_syn1
        self.src_syn2 = src_syn2
        self.src_neg = src_neg
        self.trg_syn1 = trg_syn1
        self.trg_syn2 = trg_syn2
        self.trg_neg = trg_neg

        # Trg Data
        if (self.trg_dataset != None and
            self.trg_syn1 != None and
            self.trg_syn2 != None and
            self.trg_neg !=None):
            self.trg_data = True
        else:
            self.trg_data = False

        # History
        self.history  = {'loss':[], 'dev_cosine':[], 'dev_f1':[], 'cross_f1':[],
                         'syn_cos':[], 'ant_cos':[], 'cross_syn':[], 'cross_ant':[]}

        # Do not update original embedding spaces
        self.semb.weight.requires_grad=False
        self.temb.weight.requires_grad=False

    def dump_weights(self, outfile):
        # Dump the weights to outfile
        w1 = self.m.weight.data.numpy()
        w2 = self.mp.weight.data.numpy()
        w3 = self.clf.weight.data.numpy()
        b = self.clf.bias.data.numpy()
        np.savez(outfile, w1, w2, w3, b)

    def load_weights(self, weight_file):
        # Load weights from weight_file
        f = np.load(weight_file)
        w1 = self.m.weight.data.copy_(torch.from_numpy(f['arr_0']))
        w2 = self.mp.weight.data.copy_(torch.from_numpy(f['arr_1']))
        w3 = self.clf.weight.data.copy_(torch.from_numpy(f['arr_2']))
        b = self.clf.bias.data.copy_(torch.from_numpy(f['arr_3']))

    def project(self, X, Y):
        """
        Project X and Y into shared space.
        X is a list of source words from the projection lexicon,
        and Y is the list of single word translations.
        """
        x_lookup = torch.LongTensor(np.array([self.sw2idx[w] for w in X]))
        y_lookup = torch.LongTensor(np.array([self.tw2idx[w] for w in Y]))
        x_embedd = self.semb(Variable(x_lookup))
        y_embedd = self.temb(Variable(y_lookup))
        x_proj = self.m(x_embedd)
        y_proj = self.mp(y_embedd)
        return x_proj, y_proj

    def project_one(self, x, src=True):
        """
        Project only a single list of words to the shared space.
        """
        if src:
            x_lookup = torch.LongTensor(np.array([self.sw2idx[w] for w in x]))
            x_embedd = self.semb(Variable(x_lookup))
            x_proj = self.m(x_embedd)
        else:
            x_lookup = torch.LongTensor(np.array([self.tw2idx[w] for w in x]))
            x_embedd = self.temb(Variable(x_lookup))
            x_proj = self.mp(x_embedd)
        return x_proj

    def projection_loss(self, x, y):
        """
        Find the loss between the two projected sets of translations.
        The loss is the proj_criterion.
        """

        x_proj, y_proj = self.project(x, y)

        # distance-based loss (cosine, mse)
        loss = self.proj_criterion(x_proj, y_proj)

        return loss

    def idx_vecs(self, sentence, model):
        """
        Converts a tokenized sentence to a vector
        of word indices based on the model.
        """
        sent = []
        for w in sentence:
            try:
                sent.append(model[w])
            except KeyError:
                sent.append(0)
        return torch.LongTensor(np.array(sent))

    def lookup(self, X, model):
        """
        Converts a batch of tokenized sentences
        to a matrix of word indices from model.
        """
        return [self.idx_vecs(s, model) for s in X]

    def ave_vecs(self, X, src=True):
        """
        Converts a batch of tokenized sentences into
        a matrix of averaged word embeddings. If src
        is True, it uses the sw2idx and semb to create
        the averaged representation. Otherwise, it uses
        the tw2idx and temb.
        """
        vecs = []
        #import pdb; pdb.set_trace()
        if src:
            # idxs = np.array(self.lookup(X, self.sw2idx))
            idxs = self.lookup(X, self.sw2idx)
            for i in idxs:
                vecs.append(self.semb(Variable(i)).mean(0))
        else:
            # idxs = np.array(self.lookup(X, self.tw2idx))
            idxs = self.lookup(X, self.tw2idx)
            for i in idxs:
                vecs.append(self.temb(Variable(i)).mean(0))
        return torch.stack(vecs)

    def predict(self, X, src=True):
        """
        Projects the averaged embeddings from X
        to the joint space and then uses either
        m (if src==True) or mp (if src==False)
        to predict the sentiment of X.
        """

        X = self.ave_vecs(X, src)
        if src:
            x_proj = self.m(X)
        else:
            x_proj = self.mp(X)
        out = F.softmax(self.clf(x_proj))
        return out

    def classification_loss(self, x, y, src=True):
        pred = self.predict(x, src=src)
        y = Variable(torch.from_numpy(y))
        loss = self.criterion(pred, y)
        return loss

    def full_loss(self, proj_x, proj_y, class_x, class_y,
                  alpha=.5):
        """
        Calculates the combined projection and classification loss.
        Alpha controls the amount of weight given to each loss term.
        """

        proj_loss = self.projection_loss(proj_x, proj_y)
        class_loss = self.classification_loss(class_x, class_y, src=True)
        return alpha * proj_loss + (1 - alpha) * class_loss

    def fit(self, proj_X, proj_Y,
            class_X, class_Y,
            weight_dir='models',
            batch_size=40,
            epochs=100,
            alpha=0.5):

        """
        Trains the model on the projection data (and
        source language sentiment data (class_X, class_Y).
        """

        num_batches = int(len(class_X) / batch_size)
        best_cross_f1 = 0
        num_epochs = 0

        for i in range(epochs):
            idx = 0
            num_epochs += 1

            for j in range(num_batches):
                cx = class_X[idx:idx+batch_size]
                cy = class_Y[idx:idx+batch_size]
                idx += batch_size
                self.optim.zero_grad()
                loss = self.full_loss(proj_X, proj_Y, cx, cy, alpha)
                loss.backward()
                self.optim.step()


                # check cosine distance between dev translation pairs
                xdev = self.pdataset._Xdev
                ydev = self.pdataset._ydev
                xp, yp = self.project(xdev, ydev)
                score = cos(xp, yp)

                # check source dev f1
                xdev = self.cdataset._Xdev
                ydev = self.cdataset._ydev
                xp = self.predict(xdev).data.numpy().argmax(1)
                # macro f1
                dev_f1 = macro_f1(ydev, xp)

                # check cosine distance between source sentiment synonyms
                p1 = self.project_one(self.src_syn1)
                p2 = self.project_one(self.src_syn2)
                syn_cos = cos(p1, p2)

                # check cosine distance between source sentiment antonyms
                p3 = self.project_one(self.src_syn1)
                n1 = self.project_one(self.src_neg)
                ant_cos = cos(p3, n1)

            # If there's no target data
            if not self.trg_data:

                if dev_f1 > best_cross_f1:
                    best_cross_f1 = dev_f1
                    weight_file = os.path.join(weight_dir, '{0}epochs-{1}batchsize-{2}alpha-{3:.3f}devf1'.format(num_epochs, batch_size, alpha, best_cross_f1))
                    self.dump_weights(weight_file)

                sys.stdout.write('\r epoch {0} loss: {1:.3f}  trans: {2:.3f}  src_f1: {3:.3f}  src_syn: {4:.3f}  src_ant: {5:.3f}'.format(i, loss.data.item(), score.data.item(), dev_f1, syn_cos.data.item(), ant_cos.data.item()))
                sys.stdout.flush()
                self.history['loss'].append(loss.data.item())
                self.history['dev_cosine'].append(score.data.item())
                self.history['dev_f1'].append(dev_f1)
                self.history['syn_cos'].append(syn_cos.data.item())
                self.history['ant_cos'].append(ant_cos.data.item())

            # If there's target data
            if self.trg_data:
                # check target dev f1
                crossx = self.trg_dataset._Xdev
                crossy = self.trg_dataset._ydev
                xp = self.predict(crossx, src=False).data.numpy().argmax(1)
                # macro f1
                cross_f1 = macro_f1(crossy, xp)


                if cross_f1 > best_cross_f1:
                    best_cross_f1 = cross_f1
                    weight_file = os.path.join(weight_dir, '{0}epochs-{1}batchsize-{2}alpha-{3:.3f}crossf1'.format(num_epochs, batch_size, alpha, best_cross_f1))
                    self.dump_weights(weight_file)

                # check cosine distance between target sentiment synonyms
                cp1 = self.project_one(self.trg_syn1, src=False)
                cp2 = self.project_one(self.trg_syn2, src=False)
                cross_syn_cos = cos(cp1, cp2)

                # check cosine distance between target sentiment antonyms
                cp3 = self.project_one(self.trg_syn1, src=False)
                cn1 = self.project_one(self.trg_neg, src=False)
                cross_ant_cos = cos(cp3, cn1)

                sys.stdout.write('\r epoch {0} loss: {1:.3f}  trans: {2:.3f}  src_f1: {3:.3f}  trg_f1: {4:.3f}  src_syn: {5:.3f}  src_ant: {6:.3f}  cross_syn: {7:.3f}  cross_ant: {8:.3f}'.format(
                    i, loss.data[0], score.data[0], dev_f1,
                    cross_f1, syn_cos.data[0], ant_cos.data[0],
                    cross_syn_cos.data[0], cross_ant_cos.data[0]))
                sys.stdout.flush()
                self.history['loss'].append(loss.data[0])
                self.history['dev_cosine'].append(score.data[0])
                self.history['dev_f1'].append(dev_f1)
                self.history['cross_f1'].append(cross_f1)
                self.history['syn_cos'].append(syn_cos.data[0])
                self.history['ant_cos'].append(ant_cos.data[0])
                self.history['cross_syn'].append(cross_syn_cos.data[0])
                self.history['cross_ant'].append(cross_ant_cos.data[0])

    def plot(self, title=None, outfile=None):
        """
        Plots the progression of the model. If outfile != None,
        the plot is saved to outfile.
        """

        h = self.history
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(h['dev_cosine'], label='translation_cosine')
        ax.plot(h['dev_f1'], label='source_f1', linestyle=':')
        ax.plot(h['cross_f1'], label='target_f1', linestyle=':')
        ax.plot(h['syn_cos'], label='source_synonyms', linestyle='--')
        ax.plot(h['ant_cos'], label='source_antonyms', linestyle='-.')
        ax.plot(h['cross_syn'], label='target_synonyms', linestyle='--')
        ax.plot(h['cross_ant'], label='target_antonyms', linestyle='-.')
        ax.set_ylim(-.5, 1.4)
        ax.legend(
                loc='upper center', bbox_to_anchor=(.5, 1.05),
                ncol=3, fancybox=True, shadow=True)
        if title:
            ax.title(title)
        if outfile:
            plt.savefig(outfile)
        else:
            plt.show()

    def confusion_matrix(self, X, Y, src=True):
        """
        Prints a confusion matrix for the model
        """
        pred = self.predict(X, src=src).data.numpy().argmax(1)
        cm = confusion_matrix(Y, pred, sorted(set(Y)))
        print(cm)

    def evaluate(self, X, Y, src=True, outfile=None):
        """
        Prints the accuracy, macro precision, macro recall,
        and macro F1 of the model on X. If outfile != None,
        the predictions are written to outfile.
        """

        pred = self.predict(X, src=src).data.numpy().argmax(1)
        acc = accuracy_score(Y, pred)
        prec = per_class_prec(Y, pred).mean()
        rec = per_class_prec(Y, pred).mean()
        f1 = macro_f1(Y, pred)
        if outfile:
            with open(outfile, 'w') as out:
                for i in pred:
                    out.write('{0}\n'.format(i))
        else:
            print('Test Set:')
            print('acc:  {0:.3f}\nmacro prec: {1:.3f}\nmacro rec: {2:.3f}\nmacro f1: {3:.3f}'.format(acc, prec, rec, f1))


def mse_loss(x,y):
    # mean squared error loss
    return torch.sum((x - y )**2) / x.data.shape[0]

def cosine_loss(x,y):
    c = nn.CosineSimilarity()
    return (1 - c(x,y)).mean()

def cos(x, y):
    """
    This returns the mean cosine similarity between two sets of vectors.
    """
    c = nn.CosineSimilarity()
    return c(x, y).mean()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-sl', '--source_lang',
                        help="source language: es, ca, eu, en (default: en)",
                        default='en')
    parser.add_argument('-tl', '--target_lang',
                        help="target language: es, ca, eu, en (default: es)",
                        default='es')
    parser.add_argument('-bi', '--binary',
                        help="binary or 4-class (default: True)",
                        default=True,
                        type=str2bool)
    parser.add_argument('-e', '--epochs',
                        help="training epochs (default: 200)",
                        default=200,
                        type=int)
    parser.add_argument('-a', '--alpha',
                        help="trade-off between projection and classification objectives (default: .001)",
                        default=.001,
                        type=float)
    parser.add_argument('-pl', '--proj_loss',
                        help="projection loss: mse, cosine (default: mse)",
                        default='mse')
    parser.add_argument('-bs', '--batch_size',
                        help="classification batch size (default: 50)",
                        default=20,
                        type=int)
    parser.add_argument('-sv', '--src_vecs',
                        help=" source language vectors (default: GoogleNewsVecs )",
                        default='google.txt')
    parser.add_argument('-tv', '--trg_vecs',
                        help=" target language vectors (default: SGNS on Wikipedia)",
                        default='sg-300-es.txt')
    parser.add_argument('-tr', '--trans',
                        help='translation pairs (default: Bing Liu Sentiment Lexicon Translations)',
                        default='lexicons/bingliu/en-es.txt')
    parser.add_argument('-da', '--dataset',
                        help="dataset to train and test on (default: opener_sents)",
                        default='opener_sents',)
    parser.add_argument('-sd', '--savedir',
                        help="where to dump weights during training (default: ./models)",
                        default='models/blse')
    parser.add_argument('-notarg', '--no_target_data', default=False, type=str2bool)
    args = parser.parse_args()


    # import datasets (representation will depend on final classifier)
    print('importing datasets')

    dataset = General_Dataset(os.path.join('datasets', args.source_lang, args.dataset),
                              None,
                              binary=args.binary,
                              rep=words,
                              one_hot=False)

    # Import monolingual vectors
    print('importing word embeddings')
    src_vecs = WordVecs(args.src_vecs)
    trg_vecs = WordVecs(args.trg_vecs)

     # Get sentiment synonyms and antonyms to check how they move during training
    synonyms1, synonyms2, neg = get_syn_ant(args.source_lang, src_vecs)

    # Import translation pairs
    pdataset = ProjectionDataset(args.trans, src_vecs, trg_vecs)

    if args.binary:
        output_dim = 2
        b = 'bi'
    else:
        output_dim = 4
        b = '4cls'

    if not args.no_target_data:
        cross_dataset = General_Dataset(os.path.join('datasets',
                                                     args.target_lang,
                                                     args.dataset),
                                        None,
                                        binary=args.binary,
                                        rep=words,
                                        one_hot=False)
        cross_syn1, cross_syn2, cross_neg = get_syn_ant(args.target_lang,
                                                        trg_vecs)

        # Set up model
        blse = BLSE(src_vecs, trg_vecs, pdataset, dataset, cross_dataset,
                    projection_loss=args.proj_loss,
                    output_dim=output_dim,
                    src_syn1=synonyms1, src_syn2=synonyms2, src_neg=neg,
                    trg_syn1=cross_syn1, trg_syn2=cross_syn2, trg_neg=cross_neg)

    else:
        # Set up model
        blse = BLSE(src_vecs, trg_vecs, pdataset, dataset, None,
                    projection_loss=args.proj_loss,
                    output_dim=output_dim,
                    src_syn1=synonyms1, src_syn2=synonyms2, src_neg=neg)

    # If there's no savedir, create it
    os.makedirs(args.savedir, exist_ok=True)

    # Fit model
    blse.fit(pdataset._Xtrain, pdataset._ytrain,
             dataset._Xtrain, dataset._ytrain,
             weight_dir=args.savedir,
             batch_size=args.batch_size, alpha=args.alpha, epochs=args.epochs)

    # Get best dev f1 and weights
    best_f1, best_params, best_weights = get_best_run(args.savedir)
    blse.load_weights(best_weights)
    print()
    print('Dev set')
    print('best dev f1: {0:.3f}'.format(best_f1))
    print('parameters: epochs {0} batch size {1} alpha {2}'.format(*best_params))


    if not args.no_target_data:
        # Evaluate on test set
        blse.evaluate(cross_dataset._Xtest, cross_dataset._ytest, src=False)

        blse.evaluate(cross_dataset._Xtest,
                      cross_dataset._ytest,
                      src=False,
                      outfile=os.path.join('predictions',
                                           args.target_lang,
                                           'blse','{0}-{1}-alpha{2}-epoch{3}-batch{4}.txt'.format(args.dataset,b, args.alpha, best_params[0], args.batch_size)))

        blse.confusion_matrix(cross_dataset._Xtest,
                              cross_dataset._ytest,
                              src=False)
        blse.plot()


if __name__ == '__main__':
    main()
