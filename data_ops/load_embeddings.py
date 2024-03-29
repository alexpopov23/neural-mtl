import os

import gensim
import numpy
import copy

def load_gensim(embeddings_path, binary=False):
    """Loads an embedding model with gensim

    Args:
        embeddings_path: A string, the path to the model

    Returns:
        embeddings: A list of vectors
        src2id: A dictionary, maps strings to integers in the list
        id2src: A dictionary, maps integers in the list to strings

    """
    _, extension = os.path.splitext(embeddings_path)
    if extension == ".txt":
        binary = False
    elif extension == ".bin":
        binary = True
    embeddings_model = gensim.models.KeyedVectors.load_word2vec_format(embeddings_path, binary=binary,
                                                                       datatype=numpy.float32)
    embeddings = embeddings_model.syn0
    zeros = numpy.zeros(len(embeddings[0]), dtype=numpy.float32)
    id2src = embeddings_model.index2word
    src2id = {v:(k+2) for k, v in enumerate(id2src)}
    src2id["<PAD>"] = 0
    src2id["<START>"] = 1
    embeddings = numpy.insert(embeddings, 0, copy.copy(zeros), axis=0)
    embeddings = numpy.insert(embeddings, 0, copy.copy(zeros), axis=0)
    if "UNK" not in src2id:
        if "unk" in src2id:
            src2id["<UNK>"] = src2id["unk"]
            id2src[src2id["<UNK>"]] = "<UNK>"
            del src2id["unk"]
        else:
            unk = numpy.zeros(len(embeddings[0]), dtype=numpy.float32)
            src2id["<UNK>"] = len(src2id)
            embeddings = numpy.concatenate((embeddings, [unk]))
    return embeddings, src2id, id2src