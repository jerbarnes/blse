import numpy as np


def ave_vecs(sentence, model):
    sent = np.array(np.zeros((model.vector_size)))
    sent_length = len(sentence.split())
    for w in sentence.split():
        try:
            sent += model[w]
        except KeyError:
            # TODO: implement a much better backoff strategy (Edit distance)
            sent += np.random.uniform(-.25, .25, model.vector_size)
    return sent / sent_length


def words(sentence, model):
    return sentence.split()


def getMyData(fname, label, model, representation=ave_vecs, encoding='utf8'):
    data = []
    for sent in open(fname):
        data.append((representation(sent, model), label))
    return data

