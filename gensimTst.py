# -*- coding: utf-8 -*-
"""
    @File   : gensimTst.py
    @Author : NLP_QiangShen ()
    @Time   : 2017/9/21 14:04
    @Todo   : 
"""

import gensim as gs
# from gensim import corpora, models, similarities
import logging
from collections import defaultdict
from pprint import pprint  # pretty-printer

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]

# remove common words and tokenize
stoplist = set('for a of the and to in'.split())
texts = [[word for word in text.lower().split() if word not in stoplist] for text in documents]

# remove words that appear only once

frequency = defaultdict(int)
for txt in texts:
    for word in txt:
        frequency[word] += 1

texts = [[token for token in txt if frequency[token] > 1] for txt in texts]

pprint(texts)

dictionary = gs.corpora.Dictionary(texts)
# dictionary.save('/tmp/deerwester.dict')  # store the dictionary, for future reference
print('dictionary : ', dictionary)

print('*****'*20)
print('dictionary.token2id : ', dictionary.token2id)

new_doc = "Human computer interaction"
new_vec = dictionary.doc2bow(new_doc.lower().split())# 返回文档的词袋模型向量[(词典索引,词频),]
# print('new_vec : ', new_vec)  # the word "interaction" does not appear in the dictionary and is ignored

corpus = [dictionary.doc2bow(txt) for txt in texts]
# corpora.MmCorpus.serialize('/tmp/deerwester.mm', corpus)  # store to disk, for later use
# print('corpus : ', corpus)
print('*****'*20)
for i in range(len(corpus)):
    print('txt : ',texts[i])
    print('corpus : ',corpus[i])

tfidf = gs.models.TfidfModel(corpus) # step 1 -- initialize a model

corpus_tfidf = tfidf[corpus]
for doc in corpus_tfidf:
    print(doc)

index = gs.similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=12)
sims = index[tfidf[new_vec]]
print(list(enumerate(sims)))