from __future__ import print_function
# Newsreader.py
# Matthew Stone, CS 533
# Original version: Spring 2017
# Updated: Spring 2018
# This code contains helper routines so you can get started
# writing custom text classification routines for datasets
# like the 20 newsgroup data set.
# For the key ideas about working with these functions,
# I'd recommend looking at the homework assignment text
# itself, together with the demo notebook designed to get
# you started.  This file is minimally commented,
# with just the bare bones needed to work out what's
# going on for yourself.

import os
import io
import nltk
import re
import itertools
import vocabulary
import numpy as np
import scipy
import sklearn
import random

# Useful helper functions for removing metadata
# (Metadata makes the classification problems too easy)
from sklearn.datasets.twenty_newsgroups import \
     strip_newsgroup_header as strip_newsgroup_header
from sklearn.datasets.twenty_newsgroups import \
     strip_newsgroup_quoting as strip_newsgroup_quoting
from sklearn.datasets.twenty_newsgroups import \
     strip_newsgroup_footer as strip_newsgroup_footer

def gmap(genf, arglist) :
    '''common iteration pattern where you map a function
       that turns its argument into an interator
       over a list of items and yield all the
       associated items in order'''
    for o in itertools.chain(*itertools.imap(genf, arglist)) :
        yield o
    
def istxt(f) :
    'check if a file contains relevant data for classification'
    _, e = os.path.splitext(f)
    return e == '.txt' or e == ''

def subfiles(folder) :
    'iterate over the text classification files under folder'
    here = [ os.path.join(folder, f) for f in os.listdir(folder) ]
    files = [ f for f in here if os.path.isfile(f) and istxt(f)]
    subdirs = [ f for f in here if os.path.isdir(f) ]
    for f in itertools.chain(files, gmap(subfiles, subdirs)) :
        yield f

def tokens(filename, strip_metadata=True) :
    '''limit the file to ascii characters,
    then return the ntlk tokens from the file
    if strip_metadata is true then
    strip out headers, footers and quotes''' 
    with open(filename) as f :
        text = f.read()
        if strip_metadata: 
            text = strip_newsgroup_footer(
                strip_newsgroup_quoting(
                    strip_newsgroup_header(text)))
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        for t in nltk.word_tokenize(text) :
            yield t

def all_textfile_tokens(dir, strip_metadata=True) :
    '''get all the tokens from text files
    contained under the passed directory'''
    return gmap(lambda f: tokens(f, strip_metadata), subfiles(dir))

def build_sparse_embedding(vocab, glovefile, d) :
    '''build an embedding matrix using the passed vocabulary,
    using the glove dataset stored in glovefile
    assuming word vectors in the dataset have dimension d.
    glove tokens are all lower case, so we'll try to match
    lower case, capitalized and upper case versions in the
    vocabulary.  the expectation is that most of the elements
    in the vocabulary are weird (names and other quirky tokens)
    so we'll return a scipy CSR sparse matrix.'''
    
    remaining_vocab = vocab.keyset()
    embeddings = np.zeros((len(remaining_vocab), d))
    
    with io.open(glovefile, encoding="utf-8") as glovedata :
        fileiter = glovedata.readlines()
        rows = []
        columns = []
        values = []
        
        for line in fileiter :
            line = line.replace("\n","").split(" ")
            try:
                glove_key, nums = line[0], [float(x.strip()) for x in line[1:]]
                for word in (glove_key, glove_key.capitalize(), glove_key.upper()) :
                    if word in remaining_vocab :
                        columns.append(np.arange(len(nums)))
                        rows.append(np.full(len(nums), vocab[word]))
                        values.append(np.array(nums))
                        remaining_vocab.remove(word)
            except Exception as e:
                print("{} broke. exception: {}. line: {}.".format(word, e, x))

        print("{} words were not in glove".format(len(remaining_vocab)))
        return scipy.sparse.coo_matrix((np.concatenate(values),
                                        (np.concatenate(rows),
                                         np.concatenate(columns))),
                                        shape=(len(vocab), d)).tocsr()

def save_sparse_csr(filename, array):
    'helper routine to efficiently save scipy CSR matrix'
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    'helper routine to efficiently load scipy CSR matrix'
    loader = np.load(filename)
    return scipy.sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                                   shape = loader['shape'])

class DataError(Exception) :
    '''Class for error in reading newsgroups-style classification problem
    instances from the file system.'''
    def __init__(self, message) :
        self.message = message
        
class DataManager(object) :
    '''Class for managing the mapping from newsgroup-style classification
    problem text files in the file system to feature matrix representations
    as used for classification'''
    
    def __init__(self,
                 train_dir_pos, train_dir_neg,
                 test_dir_pos, test_dir_neg,
                 make_features, get_features,
                 dev_dir_pos=None, dev_dir_neg=None,
                 strip_metadata=True) :
        self.intialized = False
        self.train_dir_pos = train_dir_pos
        self.train_dir_neg = train_dir_neg
        self.test_dir_pos = test_dir_pos
        self.test_dir_neg = test_dir_neg
        self.dev_dir_pos = dev_dir_pos
        self.dev_dir_neg = dev_dir_neg
        self.make_features = make_features
        self.get_features = get_features
        self.features = None
        self.strip_metadata = strip_metadata
        self.cache = dict()

    def initialize(self, build_cache=False) :
        '''intialize: query the file system to get the instances
        to work with and use the passed callback to compile the
        features to classify with'''
        
        self.train_files_pos = [ f for f in subfiles(self.train_dir_pos) ]
        self.train_files_neg = [ f for f in subfiles(self.train_dir_neg) ]
        test_files_pos = [ f for f in subfiles(self.test_dir_pos) ]
        test_files_neg = [ f for f in subfiles(self.test_dir_neg) ]
        if self.dev_dir_pos and self.dev_dir_neg :
            self.dev_files_pos = [f for f in subfiles(self.dev_dir_pos) ]
            self.dev_files_neg = [f for f in subfiles(self.dev_dir_neg) ]
            self.test_files_pos = test_files_pos
            self.test_files_neg = test_files_neg
        else :
            dev_len = int(len(test_files_pos) / 4)
            self.dev_files_pos = test_files_pos[:dev_len]
            self.dev_files_neg = test_files_neg[:dev_len]
            self.test_files_pos = test_files_pos[dev_len:]
            self.test_files_neg = test_files_neg[dev_len:]
        self.initialized = True
        self.features = self.make_features(self)
        if build_cache :
            self.training_data(shuffle=False)
            self.dev_data(shuffle=False)
            self.test_data(shuffle=False)
            
    def all_train_tokens(self) :
        '''iterate over the training tokens associated with the passed data set'''
        
        if self.initialized :
            return gmap(lambda f : tokens(f, self.strip_metadata),
                        self.train_files_pos + self.train_files_neg)
        else:
            raise DataError("Must call initialize() on Data Manager before accessing data")

    def _files_to_features(self, files) :
        '''helper function for compiling a feature matrix from a list of files.
        each row in the matrix corresponds to one of the files.
        the matrix is returned in sparse CSR representation.
        the passed callback is used to get the feature representation
        from the file tokens'''

        row = 0
        rows = []
        columns = []
        values = []
        for f in files:
            l = np.array([i for i in self.get_features(self.features,
                                                       tokens(f, self.strip_metadata))])
            columns.append(l)
            rows.append(np.full(l.shape, row, dtype=np.int))
            values.append(np.full(l.shape, 1., dtype=np.float))
            row = row + 1
        cd = np.concatenate(columns)
        return scipy.sparse.coo_matrix((np.concatenate(values), 
                                        (np.concatenate(rows), 
                                        cd)),  
                                        shape=(row,len(self.features))).tocsr()

    def _get_data(self, files_pos, files_neg, cache_key=None, shuffle=True) :
        '''helper function for returning data from a particular class.
        this is where we shuffle the data (if requested)
        to avoid glitches from (e.g.) the order of training elements.'''

        if cache_key and cache_key in self.cache :
            X, y = self.cache[cache_key]
        else :
            X = self._files_to_features(files_pos + files_neg)
            y = np.concatenate([np.ones_like(files_pos, dtype=np.float),
                                np.zeros_like(files_neg, dtype=np.float)])
            if cache_key :
                self.cache[cache_key] = (X, y)
        if shuffle :
            perm = np.random.permutation(X.shape[0])
            X = X[perm]
            y = y[perm]
        return X, y

    def training_data(self, shuffle=True) :
        '''get matrices for the training data and training categories from the collection'''
        if self.initialized :
            return self._get_data(self.train_files_pos, self.train_files_neg,
                                  'train', shuffle)
        else:
            raise DataError("Must call initialize() on Data Manager before accessing data")

    def dev_data(self, shuffle=True) :
        '''get matrices for the development data and development categories from the collection'''
        if self.initialized :
            return self._get_data(self.dev_files_pos, self.dev_files_neg,
                                  'dev', shuffle)
        else:
            raise DataError("Must call initialize() on Data Manager before accessing data")

    def test_data(self, shuffle=True) :
        '''get matrices for the test data and test categories from the collection'''
        if self.initialized :
            return self._get_data(self.test_files_pos, self.test_files_neg,
                                  'test', shuffle)
        else:
            raise DataError("Must call initialize() on Data Manager before accessing data")
    

