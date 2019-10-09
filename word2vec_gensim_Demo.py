# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 01:02:40 2019

@author: Phuc123
"""

import bs4 as bs
import urllib.request
import re
import nltk

scrapped_data = urllib.request.urlopen('https://en.wikipedia.org/wiki/Machine_learning')
article = scrapped_data .read()

parsed_article = bs.BeautifulSoup(article,'lxml')

paragraphs = parsed_article.find_all('p')

article_text = ""
for p in paragraphs:
    article_text += p.text
# Cleaing the text
processed_article = article_text.lower()
processed_article = re.sub('[^a-zA-Z]', ' ', processed_article )
processed_article = re.sub(r'\s+', ' ', processed_article)
processed_article=("king brave man")
# Preparing the dataset
nltk.download('punkt')
all_sentences = nltk.sent_tokenize(processed_article)

all_words = [nltk.word_tokenize(sent) for sent in all_sentences]

# Removing Stop Words
nltk.download('stopwords')
from nltk.corpus import stopwords
for i in range(len(all_words)):
    all_words[i] = [w for w in all_words[i] if w not in stopwords.words('english')]

from gensim.models import Word2Vec
word2vec = Word2Vec(all_words, min_count=1)
vocabulary = word2vec.wv.vocab
print(vocabulary)
v1 = word2vec.wv['man']
sim_words = word2vec.wv.most_similar('man')