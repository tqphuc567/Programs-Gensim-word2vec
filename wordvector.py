# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 08:37:56 2019

@author: Phuc123
"""
# Tao vector tu
>>> from gensim.test.utils import common_texts, get_tmpfile
>>> from gensim.models import Word2Vec
>>> common_texts= "the dog saw a cat. the dog chased the cat . the cat climbed a tree"
>>> model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)
>>> word_vectors = model.wv
>>> vocabulary = model.wv.vocab
>>> v1=model.wv['c']
>>> path=get_tmpfile("model.wv")
>>> model.save("model.wv")
>>> print(model.wv.index2entity[:10])
>>> print([model.wv.vocab[word].count for word in model.wv.index2entity[:10]])
# Tao vector tu trong model co san
>>> from gensim.test.utils import get_tmpfile
>>> from gensim.models import KeyedVectors
>>> fname = get_tmpfile("vectors.kv")
>>> word_vectors.save(fname)
>>> word_vectors = KeyedVectors.load(fname, mmap='r')
>>> print(word_vectors.wv.index2entity[:10])
# Khoi tao word vector tu mot tep hien co tren dia o dinh dang goc Google new viet tren C nhu KeyedVector
>>> from gensim.test.utils import datapath
>>> wv_from_text = KeyedVectors.load_word2vec_format(datapath('word2vec_pre_kv_c'), binary=False)  # C text format
>>> wv_from_bin = KeyedVectors.load_word2vec_format(datapath("euclidean_vectors.bin"), binary=True)  # C bin format
>>> print (wv_from_bin.index2entity[:10])
>>> print (wv_from_text.index2entity[:10])
