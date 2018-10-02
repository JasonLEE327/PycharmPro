from __future__ import print_function
import os
if '/Users/zhaoleiwang/Desktop/NLP/data' != os.getcwd():
    os.chdir('/Users/zhaoleiwang/Desktop/NLP/mypro')

import nltk
import re
import itertools
import vocabulary
import newsreader
import numpy as np
import scipy
import sklearn
from sklearn.preprocessing import StandardScaler
try:
    import cPickle as pickle
except:
    import pickle

vocab_file, vocab_file_type = "reviews-vocab.pkl", "pickle"

embedding_file, embedding_dimensions, embedding_cache = \
    "glove.6B.50d.txt", 50, "reviews-embedding.npz"

all_data, train_dir, dev_dir, test_dir = \
    "reviews", "reviews/train/", "reviews/dev/", "reviews/test/"

class1, class2 = "pos", "neg"

has_bad_metadata = False

made_vocabulary = True
if made_vocabulary :
    v = vocabulary.Vocabulary.load(vocab_file, file_type=vocab_file_type)
else:
    tokens = newsreader.all_textfile_tokens(all_data, strip_metadata=has_bad_metadata)
    v = vocabulary.Vocabulary.from_iterable(tokens, file_type=vocab_file_type)
    v.save(vocab_file)
v.stop_growth()

made_embedding = True
if made_embedding :
    e = newsreader.load_sparse_csr(embedding_cache)
else:
    e = newsreader.build_sparse_embedding(v, embedding_file, embedding_dimensions)
    newsreader.save_sparse_csr(embedding_cache, e)