# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 02:52:16 2019

@author: Phuc123
"""
from __future__ import division  # py3 "true division"

import logging
import sys
import os
import heapq
from timeit import default_timer
from copy import deepcopy
from collections import defaultdict
import threading
import itertools
import warnings

from gensim.utils import keep_vocab_item, call_on_class_only
from gensim.models.keyedvectors import Vocab, Word2VecKeyedVectors
from gensim.models.base_any2vec import BaseWordEmbeddingsModel

try:
    from queue import Queue, Empty
except ImportError:
    from Queue import Queue, Empty

from numpy import exp, dot, zeros, random, dtype, float32 as REAL,\
    uint32, seterr, array, uint8, vstack, fromstring, sqrt,\
    empty, sum as np_sum, ones, logaddexp, log, outer

from scipy.special import expit

from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from gensim.utils import deprecated
from six import iteritems, itervalues, string_types
from six.moves import range

logger = logging.getLogger(__name__)

try:
    from gensim.models.word2vec_inner import train_batch_sg, train_batch_cbow
    from gensim.models.word2vec_inner import score_sentence_sg, score_sentence_cbow
    from gensim.models.word2vec_inner import FAST_VERSION, MAX_WORDS_IN_BATCH

except ImportError:
    # failed... fall back to plain numpy (20-80x slower training than the above)
    FAST_VERSION = -1
    MAX_WORDS_IN_BATCH = 10000


def train_batch_cbow(model, sentences, alpha, work=None, neu1=None, compute_loss=False):
        """Update CBOW model by training on a sequence of sentences.
        Called internally from :meth:`~gensim.models.word2vec.Word2Vec.train`.
        Warnings
        --------
        This is the non-optimized, pure Python version. If you have a C compiler, Gensim
        will use an optimized code path from :mod:`gensim.models.word2vec_inner` instead.
        Parameters
        ----------
        model : :class:`~gensim.models.word2vec.Word2Vec`
            The Word2Vec model instance to train.
        sentences : iterable of list of str
            The corpus used to train the model.
        alpha : float
            The learning rate
        work : object, optional
            Unused.
        neu1 : object, optional
            Unused.
        compute_loss : bool, optional
            Whether or not the training loss should be computed in this batch.
        Returns
        -------
        int
            Number of words in the vocabulary actually used for training (that already existed in the vocabulary
            and were not discarded by negative sampling).
        """
        result = 0
        for sentence in sentences:
            word_vocabs = [
                model.wv.vocab[w] for w in sentence if w in model.wv.vocab
                and model.wv.vocab[w].sample_int > model.random.rand() * 2 ** 32
            ]
            for pos, word in enumerate(word_vocabs):
                reduced_window = model.random.randint(model.window)  # `b` in the original word2vec code
                start = max(0, pos - model.window + reduced_window)
                window_pos = enumerate(word_vocabs[start:(pos + model.window + 1 - reduced_window)], start)
                word2_indices = [word2.index for pos2, word2 in window_pos if (word2 is not None and pos2 != pos)]
                l1 = np_sum(model.wv.syn0[word2_indices], axis=0)  # 1 x vector_size
                if word2_indices and model.cbow_mean:
                    l1 /= len(word2_indices)
                train_cbow_pair(model, word, word2_indices, l1, alpha, compute_loss=compute_loss)
            result += len(word_vocabs)
        return result
    
def train_cbow_pair(model, word, input_word_indices, l1, alpha, learn_vectors=True, learn_hidden=True,
                    compute_loss=False, context_vectors=None, context_locks=None, is_ft=False):
    """Train the passed model instance on a word and its context, using the CBOW algorithm.
    Parameters
    ----------
    model : :class:`~gensim.models.word2vec.Word2Vec`
        The model to be trained.
    word : str
        The label (predicted) word.
    input_word_indices : list of int
        The vocabulary indices of the words in the context.
    l1 : list of float
        Vector representation of the label word.
    alpha : float
        Learning rate.
    learn_vectors : bool, optional
        Whether the vectors should be updated.
    learn_hidden : bool, optional
        Whether the weights of the hidden layer should be updated.
    compute_loss : bool, optional
        Whether or not the training loss should be computed.
    context_vectors : list of list of float, optional
        Vector representations of the words in the context. If None, these will be retrieved from the model.
    context_locks : list of float, optional
        The lock factors for each word in the context.
    is_ft : bool, optional
        If True, weights will be computed using `model.wv.syn0_vocab` and `model.wv.syn0_ngrams`
        instead of `model.wv.syn0`.
    Returns
    -------
    numpy.ndarray
        Error vector to be back-propagated.
    """
    if context_vectors is None:
        if is_ft:
            context_vectors_vocab = model.wv.syn0_vocab
            context_vectors_ngrams = model.wv.syn0_ngrams
        else:
            context_vectors = model.wv.syn0
    if context_locks is None:
        if is_ft:
            context_locks_vocab = model.syn0_vocab_lockf
            context_locks_ngrams = model.syn0_ngrams_lockf
        else:
            context_locks = model.syn0_lockf

    neu1e = zeros(l1.shape)

    if model.hs:
        l2a = model.syn1[word.point]  # 2d matrix, codelen x layer1_size
        prod_term = dot(l1, l2a.T)
        fa = expit(prod_term)  # propagate hidden -> output
        ga = (1. - word.code - fa) * alpha  # vector of error gradients multiplied by the learning rate
        if learn_hidden:
            model.syn1[word.point] += outer(ga, l1)  # learn hidden -> output
        neu1e += dot(ga, l2a)  # save error

        # loss component corresponding to hierarchical softmax
        if compute_loss:
            sgn = (-1.0) ** word.code  # ch function, 0-> 1, 1 -> -1
            model.running_training_loss += sum(-log(expit(-sgn * prod_term)))

    if model.negative:
        # use this word (label = 1) + `negative` other random words not from this sentence (label = 0)
        word_indices = [word.index]
        while len(word_indices) < model.negative + 1:
            w = model.cum_table.searchsorted(model.random.randint(model.cum_table[-1]))
            if w != word.index:
                word_indices.append(w)
        l2b = model.syn1neg[word_indices]  # 2d matrix, k+1 x layer1_size
        prod_term = dot(l1, l2b.T)
        fb = expit(prod_term)  # propagate hidden -> output
        gb = (model.neg_labels - fb) * alpha  # vector of error gradients multiplied by the learning rate
        if learn_hidden:
            model.syn1neg[word_indices] += outer(gb, l1)  # learn hidden -> output
        neu1e += dot(gb, l2b)  # save error

        # loss component corresponding to negative sampling
        if compute_loss:
            model.running_training_loss -= sum(log(expit(-1 * prod_term[1:])))  # for the sampled words
            model.running_training_loss -= log(expit(prod_term[0]))  # for the output word

    if learn_vectors:
        # learn input -> hidden, here for all words in the window separately
        if is_ft:
            if not model.cbow_mean and input_word_indices:
                neu1e /= (len(input_word_indices[0]) + len(input_word_indices[1]))
            for i in input_word_indices[0]:
                context_vectors_vocab[i] += neu1e * context_locks_vocab[i]
            for i in input_word_indices[1]:
                context_vectors_ngrams[i] += neu1e * context_locks_ngrams[i]
        else:
            if not model.cbow_mean and input_word_indices:
                neu1e /= len(input_word_indices)
            for i in input_word_indices:
                context_vectors[i] += neu1e * context_locks[i]
    return neu1e

def train_sg_pair(model, word, context_index, alpha, learn_vectors=True, learn_hidden=True,
                  context_vectors=None, context_locks=None, compute_loss=False, is_ft=False):
    """Train the passed model instance on a word and its context, using the Skip-gram algorithm.
    Parameters
    ----------
    model : :class:`~gensim.models.word2vec.Word2Vec`
        The model to be trained.
    word : str
        The label (predicted) word.
    context_index : list of int
        The vocabulary indices of the words in the context.
    alpha : float
        Learning rate.
    learn_vectors : bool, optional
        Whether the vectors should be updated.
    learn_hidden : bool, optional
        Whether the weights of the hidden layer should be updated.
    context_vectors : list of list of float, optional
        Vector representations of the words in the context. If None, these will be retrieved from the model.
    context_locks : list of float, optional
        The lock factors for each word in the context.
    compute_loss : bool, optional
        Whether or not the training loss should be computed.
    is_ft : bool, optional
        If True, weights will be computed using `model.wv.syn0_vocab` and `model.wv.syn0_ngrams`
        instead of `model.wv.syn0`.
    Returns
    -------
    numpy.ndarray
        Error vector to be back-propagated.
    """
    if context_vectors is None:
        if is_ft:
            context_vectors_vocab = model.wv.syn0_vocab
            context_vectors_ngrams = model.wv.syn0_ngrams
        else:
            context_vectors = model.wv.syn0
    if context_locks is None:
        if is_ft:
            context_locks_vocab = model.syn0_vocab_lockf
            context_locks_ngrams = model.syn0_ngrams_lockf
        else:
            context_locks = model.syn0_lockf

    if word not in model.wv.vocab:
        return
    predict_word = model.wv.vocab[word]  # target word (NN output)

    if is_ft:
        l1_vocab = context_vectors_vocab[context_index[0]]
        l1_ngrams = np_sum(context_vectors_ngrams[context_index[1:]], axis=0)
        if context_index:
            l1 = np_sum([l1_vocab, l1_ngrams], axis=0) / len(context_index)
    else:
        l1 = context_vectors[context_index]  # input word (NN input/projection layer)
        lock_factor = context_locks[context_index]

    neu1e = zeros(l1.shape)

    if model.hs:
        # work on the entire tree at once, to push as much work into numpy's C routines as possible (performance)
        l2a = deepcopy(model.syn1[predict_word.point])  # 2d matrix, codelen x layer1_size
        prod_term = dot(l1, l2a.T)
        fa = expit(prod_term)  # propagate hidden -> output
        ga = (1 - predict_word.code - fa) * alpha  # vector of error gradients multiplied by the learning rate
        if learn_hidden:
            model.syn1[predict_word.point] += outer(ga, l1)  # learn hidden -> output
        neu1e += dot(ga, l2a)  # save error

        # loss component corresponding to hierarchical softmax
        if compute_loss:
            sgn = (-1.0) ** predict_word.code  # `ch` function, 0 -> 1, 1 -> -1
            lprob = -log(expit(-sgn * prod_term))
            model.running_training_loss += sum(lprob)

    if model.negative:
        # use this word (label = 1) + `negative` other random words not from this sentence (label = 0)
        word_indices = [predict_word.index]
        while len(word_indices) < model.negative + 1:
            w = model.cum_table.searchsorted(model.random.randint(model.cum_table[-1]))
            if w != predict_word.index:
                word_indices.append(w)
        l2b = model.syn1neg[word_indices]  # 2d matrix, k+1 x layer1_size
        prod_term = dot(l1, l2b.T)
        fb = expit(prod_term)  # propagate hidden -> output
        gb = (model.neg_labels - fb) * alpha  # vector of error gradients multiplied by the learning rate
        if learn_hidden:
            model.syn1neg[word_indices] += outer(gb, l1)  # learn hidden -> output
        neu1e += dot(gb, l2b)  # save error

        # loss component corresponding to negative sampling
        if compute_loss:
            model.running_training_loss -= sum(log(expit(-1 * prod_term[1:])))  # for the sampled words
            model.running_training_loss -= log(expit(prod_term[0]))  # for the output word

    if learn_vectors:
        if is_ft:
            model.wv.syn0_vocab[context_index[0]] += neu1e * context_locks_vocab[context_index[0]]
            for i in context_index[1:]:
                model.wv.syn0_ngrams[i] += neu1e * context_locks_ngrams[i]
        else:
            l1 += neu1e * lock_factor  # learn input -> hidden (mutates model.wv.syn0[word2.index], if that is l1)
    return neu1e

from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec

common_texts= "the dog saw a cat.the dog chased the cat. the cat climbed a tree"

model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)
vocabulary = model.wv.vocab


sentences="the dog saw a cat.the dog chased the cat. the cat climbed a tree"
    
train_batch_cbow(model, sentences, 3/4, work=None, neu1=None, compute_loss=False)