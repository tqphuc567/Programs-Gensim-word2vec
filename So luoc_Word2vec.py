# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 15:20:46 2019

@author: Phuc123
"""
# 1.Tính sigmoid probability
import math
z = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]
z_exp = [math.exp(i) for i in z]
print([round(i, 2) for i in z_exp])

math.exp(1)

sum_z_exp = sum(z_exp)
print(round(sum_z_exp, 2))

softmax = [i / sum_z_exp for i in z_exp]
print([round(i, 3) for i in softmax])

# # Ví dụ dùng Numpy
import numpy as np
z = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]
softmax = np.exp(z)/np.sum(np.exp(z))
softmax

# EX. Program to multiply two matrices using nested loops
# Give Paragraghs= “the dog saw a cat”, “the dog chased the cat”, “the cat climbed a tree”
# Hoi? Tim hieu moi quan he giua tu "cat" va "climbed".Voi input la "cat" va Output ~ Target "climbed"
# -> B1: Order index corpus vocabulary=(a, cat, chased, climbed, dog, saw, the, tree)
# -> B2:Chon lop an Hidden layer 3 neuros
# -> B3:Tinh Hidden layer neurons 
# Khoi tao ma tran ngan nhien dau vao cua lop an (WI) la 8x3 matrix voi cac tham so sau:
WI = [[-0.094491,-0.443977,0.33917],
     [-0.490796,-0.229903,0.065460],
     [0.072921,0.172246,-0.357751],
     [0.104514,-0.463000,0.079367],
     [-0.226080,-0.154659,-0.038422],
     [0.406115,-0.192794,-0.441992],
     [0.181755,0.088268,0.277574],
     [-0.055334,0.491792,0.263102]]
# Khoi tao ma tran ngan nhien dau ra cua lop an (WO) la 3x8 matrix voi cac tham so sau:
WO = [[0.023074,0.479901,0.432148,0.375480,-0.364732,-0.119840,0.266070,-0.351000],
     [-0.368008,0.424778,-0.257104,-0.148817,0.033922,0.353874,-0.114942,0.130904],
     [0.422434,0.364503,0.467865,-0.020302,-0.423890,-0.438777,0.268529,-0.446787]]

# Tinh dau vao Hidden layer neurons (3 neurons) theo cong thuc: H^t=X^t.WI, ket qua la:
H =[[-0.490796,-0.229903,0.065460]] # lay giong input cua one hot vector "car"=[0,1,0,0,0,0,0,0]

# Tinh dau ra cua Hidden layer theo cong thuc: H^t.WO

result = [[0,0,0,0,0,0,0,0]]

# iterate through rows of X
for i in range(len(H)):
   # iterate through columns of Y
   for j in range(len(WO[0])):
       # iterate through rows of Y
       for k in range(len(WO)):
           result[i][j] += H[i][k] * WO[k][j]

for r in result:
   print(r)
   
# EX.Tính sigmoid probability # Van dung
import math
z = [0.100934, -0.309331 , -0.122361 , -0.151399, 0.143463, -0.051262, -0.079686, 0.112928]
z_exp = [math.exp(i) for i in z]
print([round(i, 2) for i in z_exp])

math.exp(1)

sum_z_exp = sum(z_exp)
print(round(sum_z_exp, 2))

softmax = [i / sum_z_exp for i in z_exp]
print([round(i, 3) for i in softmax])

# # Ví dụ dùng Numpy
import numpy as np
z = [0.100934, -0.309331 , -0.122361 , -0.151399, 0.143463, -0.051262, -0.079686, 0.112928]
softmax = np.exp(z)/np.sum(np.exp(z))
softmax

# B4: Tinh ty le loi error bang bang cach lay vector Target - vector SoftMax - Subtract Two Matrices
matrix1 = [[0,0,0,1,0,0,0,0]]
softmax = [[0.1430733 , 0.09492547, 0.11444131, 0.11116595, 0.14928931,
       0.1228742 , 0.1194308 , 0.14479966]]
rmatrix = [[0,0,0,0,0,0,0,0]]
for i in range(len(matrix1)):
    for j in range(len(matrix1[0])):
        rmatrix[i][j] = matrix1[i][j] - softmax[i][j]
for r in rmatrix:
    print(r)

# B5:Updated using backpropagation (Cap nhat bang cach su dung lan truyen nguoc khi biet loi error)
# B6:Update cho den khi ti le loi error la nho nhat (Min) thi dung lai -> result

# 2. Cai dat thuan toan Word2vec Model dung thu vien gensim
import bs4 as bs
import urllib.request
import re
import nltk
import string

article_text="the dog saw a cat. the dog chased the cat . the cat climbed a tree"

# Cleaing the text
processed_article = article_text.lower()
processed_article = re.sub('[^a-zA-Z]', ' ', processed_article )
processed_article = re.sub(r'\s+', ' ', processed_article)

# Preparing the dataset
import collections
all_words=processed_article.split()
word_counts=collections.Counter(all_words)
all=[[words for words in sorted(word_counts)]]
    
# Python program to sort a list of strings 
#vocabulary.sort()
# Using sort() function 
#vocabulary.sort() 
#print(vocabulary)

num_features = 3  # Word vector dimensionality # So chieu cua vector tu
min_word_count = 0 # Minimum word count # So tu xuat hien toi thieu
num_workers = 2     # Number of parallel threads #
context = 3       # Context window size # Kich thuoc cua so ngu canh
downsampling = 1e-3 # (0.001) Downsample setting for frequent words # Cai dat mau cho cac tu thuong xuyen
from gensim.models import Word2Vec
word2vec = Word2Vec(all,\
                          workers=num_workers,\
                          size=num_features,\
                          min_count=min_word_count,\
                          window=context,
                          sample=downsampling)
vocabulary = word2vec.wv.vocab
print(vocabulary)
v1 = word2vec.wv['cat']
v2 = word2vec.wv['climbed']
sim_words = word2vec.wv.most_similar('climbed')

article_text="the dog saw a cat. the dog chased the cat . the cat climbed a tree"

word2vec.similarity("climbed", "cat")
word2vec.most_similar("dog")


# 2. Dự đoán probability của từ:
## Dùng gensim cho word2vec:
num_features = 10  # Word vector dimensionality # So chieu cua vector tu
min_word_count = 0 # Minimum word count # So tu xuat hien toi thieu
num_workers = 4     # Number of parallel threads #
context = 10        # Context window size # Kich thuoc cua so ngu canh
downsampling = 1e-3 # (0.001) Downsample setting for frequent words # Cai dat mau cho cac tu thuong xuyen

# Initializing the train model # Khoi tao mo hinh dao tao
from gensim.models import word2vec
sentences="the dog saw a cat.the dog chased the cat. the cat climbed a tree"
print("Training model....", len(sentences))
model = word2vec.Word2Vec(sentences,\
                          workers=num_workers,\
                          size=num_features,\
                          min_count=min_word_count,\
                          window=context,
                          sample=downsampling)

# To make the model memory efficient
model.init_sims(replace=True)

# Saving the model for later use. Can be loaded using Word2Vec.load()
model_name = "300features_40minwords_10context"
model.save(model_name)

vocabulary = model.wv.vocab
print(vocabulary)

v1 = model.wv['n']
model.wv.most_similar("k")