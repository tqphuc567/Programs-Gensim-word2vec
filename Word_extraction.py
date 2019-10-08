# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 09:41:24 2019

@author: Phuc123
"""
import bs4 as bs
import urllib.request
import re
import nltk
import math
from nltk import ngrams
file=open('myfile.txt','r')
txt = file.read()
n = 4
fourgrams = ngrams(txt.split(), n)
for grams in fourgrams:
  print (grams)
  
words = 'This is is random text we’re going to split apart'
x=[]
for word in words.split():
    x.append(word)
    if len(x) == 4:
        print(x)
        x=[]
print(x)
  
pwd()
# Cach 1Ham cac tu khong trung nhau 
import collections

sentence = """the dog saw a cat the dog chased the cat the cat climbed a tree"""

def count(sentence):
    words = sentence.split()
    word_counts = collections.Counter(words)
    for word, count in sorted(word_counts.items()):
        print('"%s" is repeated %d time%s.' % (word, count, "s" if count > 1 else ""))
        
all_words=count(sentence)
#Cach 2 Ham trich xuat cac tu khong trung nhau
from itertools import groupby 

mysentence = ("As far as the laws of mathematics refer to reality "
              "they are not certain as far as they are certain "
              "they do not refer to reality")
words = mysentence.split() # get a list of whitespace-separated words
for word, duplicates in groupby(sorted(words)): # sort and group duplicates
    count = len(list(duplicates)) # count how many times the word occurs
    print('"{word}" is repeated {count} time{s}'.format(
            word=word, count=count,  s='s'*(count > 1)))

# Cach 3: Ham trich xuat cac tu khong trung nhau

x = "As far as the laws of mathematics refer to reality they are not certain as far as they are certain they do not refer to reality"
words = x.split(" ")
words.sort()

last_word = ""
for word in words:
    if word != last_word:
        count = [i for i, w in enumerate(words) if w == word]
        print(word + " is repeated " + str(len(count)) + " times.")
    last_word = word
            