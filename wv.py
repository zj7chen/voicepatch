#!/usr/bin/env python3

from __future__ import print_function
import pickle
import time
import numpy as np
from collections import Counter
from gensim.models import word2vec, KeyedVectors

from os import listdir
import re
import codecs


def refine(data):
    words = re.findall("[a-zA-Z'-]+", data)
    words = ["".join(word.split("'")) for word in words]
    # words = ["".join(word.split("-")) for word in words]
    data = ' '.join(words)
    return data


# parse training data
print("Parsing training data...")
ts = time.time()
training_data = []
with codecs.open('model/all_words.txt', "w", encoding='utf-8', errors='ignore') as out:
    for filename in listdir('data/Holmes_Training_Data'):
        # with open('data/Holmes_Training_Data/%s' % filename, 'r') as f:
        with codecs.open('data/Holmes_Training_Data/%s' % filename, "r", encoding='utf-8', errors='ignore') as f:
            parseFlag = False
            data = ""
            for line in f:
                if line.find('*END*THE SMALL PRINT!') > -1 \
                    or line.find('ENDTHE SMALL PRINT!') > -1:  # remove some useless paragraphs at the beginning
                    parseFlag = True
                    continue
                if parseFlag:
                    line = line.replace("\r\n", " ")
                    data += line.lower()
            sentences = data.split('.')
            training_data += [refine(sentence).split() for sentence in sentences]
            data = refine(data)
            out.write(data + ' ')
pickle.dump(training_data, open('data/training_data', 'wb'), True)
print("Time Elapsed: {0} secs\n".format(time.time() - ts))


# build word vector, type: word embedding
WORD_VECTOR_SIZE = 300
print("Building embedded word vector...")
ts = time.time()
corpus = word2vec.Text8Corpus("model/all_words.txt")
word_vector = word2vec.Word2Vec(corpus, size=WORD_VECTOR_SIZE)
word_vector.wv.save_word2vec_format(u"model/word_vector.txt", binary=False)
word_vector.wv.save_word2vec_format(u"model/word_vector.bin", binary=True)
print("Time Elapsed: {0} secs\n".format(time.time() - ts))
