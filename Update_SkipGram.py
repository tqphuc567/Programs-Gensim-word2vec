# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 02:03:04 2019

@author: Phuc123
"""

def train_batch_sg(model, sentences, alpha, work=None, compute_loss=False):
        """Update skip-gram model by training on a sequence of sentences.
        Called internally from :meth:`~gensim.models.word2vec.Word2Vec.train`.
        Warnings
        --------
        This is the non-optimized, pure Python version. If you have a C compiler, Gensim
        will use an optimized code path from :mod:`gensim.models.word2vec_inner` instead.
        Parameters
        ----------
        model : :class:`~gensim.models.word2Vec.Word2Vec`
            The Word2Vec model instance to train.
        sentences : iterable of list of str
            The corpus used to train the model.
        alpha : float
            The learning rate
        work : object, optional
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
            word_vocabs = [model.wv.vocab[w] for w in sentence if w in model.wv.vocab
                           and model.wv.vocab[w].sample_int > model.random.rand() * 2 ** 32]
            for pos, word in enumerate(word_vocabs):
                reduced_window = model.random.randint(model.window)  # `b` in the original word2vec code

                # now go over all words from the (reduced) window, predicting each one in turn
                start = max(0, pos - model.window + reduced_window)
                for pos2, word2 in enumerate(word_vocabs[start:(pos + model.window + 1 - reduced_window)], start):
                    # don't train on the `word` itself
                    if pos2 != pos:
                        train_sg_pair(
                            model, model.wv.index2word[word.index], word2.index, alpha, compute_loss=compute_loss
                        ) 

            result += len(word_vocabs)
        return result
    
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
path = get_tmpfile("word2vec.model")  

common_texts= "the dog saw a cat.the dog chased the cat. the cat climbed a tree"

model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)
vocabulary = model.wv.vocab


sentences="the dog saw a cat.the dog chased the cat. the cat climbed a tree"
    
train_batch_sg(model, sentences,3/4, work=None, compute_loss=False)