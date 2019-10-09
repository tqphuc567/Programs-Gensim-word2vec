# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 03:48:12 2019

@author: Phuc123
"""

import math
import re
import nltk
import gensim
# getting the training vocabulary & word frequency (lay tu file du lieu nhi phan (.bin))
from gensim.models.KeyedVectors import KeyedVectors
    # load a binary-format word2vec vector set
    wv = KeyedVectors.load_word2vec_format('C:/Users/Phuc123/GoogleNews-vectors-negative300.bin', binary=True) 
    # print number of words included
    print(len(wv))
    # print 1st 20 words, in order they appeared in file
    print(wv.index2entity[:100])
    # Tap tu dung Google new
    vocab = wv.vocab.keys()
    # chuyen mot tu thanh vector
    v1 = wv['good']
    # Do chieu dai tap hop tu dung
    wordsInVocab = len(vocab)
    # Tinh do tuong tu 
    wv.most_similar('good')
    print (wordsInVocab)
    print (wv.similarity('this', 'is'))
    print (wv.similarity('post', 'book'))
    
#getting the training vocabulary & word frequency (du lieu lay mo hinh dao tap truoc do word2vec.model )
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
path = get_tmpfile("word2vec.model")
common_texts= "the dog saw a cat. the dog chased the cat . the cat climbed a tree"
model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)
model.save("word2vec.model")

from gensim.models.word2vec import Word2Vec
    w2v_model = Word2Vec.load("word2vec.model")
    # print number of words included
    print(len(w2v_model.wv))
    # print 1st 10 words, in model's internal order
    print(w2v_model.wv.index2entity[:10])
    # print counts of 1st 10 words as seen during vocabulary-discovery
    print([w2v_model.wv.vocab[word].count for word in w2v_model.wv.index2entity[:10]])
    
#Convert word2vec bin file to text
from gensim.models.keyedvectors import KeyedVectors
model = KeyedVectors.load_word2vec_format('path/to/GoogleNews-vectors-negative300.bin', binary=True)
wv.save_word2vec_format('C:/Users/Phuc123/GoogleNews-vectors-negative300.txt', binary=False)

