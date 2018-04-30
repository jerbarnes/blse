import numpy as np


def ave_vecs(sentence, model):
    sent = np.array(np.zeros((model.vector_size)))
    sent_length = len(sentence.split())
    for w in sentence.split():
        try:
            sent += model[w]
        except:
            # TODO: implement a much better backoff strategy (Edit distance)
            sent += np.random.uniform(-.25, .25, model.vector_size)
    return sent / sent_length


def idx_vecs(sentence, model):
    """Returns a list of vectors of the tokens
    in the sentence if they are in the model."""
    sent = []
    for w in sentence.split():
        try:
            sent.append(model[w])
        except:
            # TODO: implement a much better backoff strategy (Edit distance)
            sent.append(model['of'])
    return sent


def bow(sentence, w2idx):
    """
    Bag of words representation
    """
    array = np.zeros(len(w2idx))
    for w in sentence:
        try:
            array[w2idx[w]] += 1
        except KeyError:
            pass
    return array

def words(sentence, model):
    return sentence.split()


def getMyData(fname, label, model, representation=ave_vecs, encoding='utf8'):
    data = []
    for sent in open(fname):
        data.append((representation(sent, model), label))
    return data

