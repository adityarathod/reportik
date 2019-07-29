import os
import pickle
from math import floor

import gensim.downloader as api
import numpy as np
from tensorflow import keras


class DataManager:
    train_documents = None
    test_documents = None
    val_documents = None
    document_tokenizer: keras.preprocessing.text.Tokenizer = None
    train_summaries = None
    test_summaries = None
    val_summaries = None
    summary_tokenizer: keras.preprocessing.text.Tokenizer = None
    summ_emb_path = None
    val_split = None
    embeddings = None
    embedding_size = None

    def __init__(self,
                 saved_dir='../data',
                 train_data_filename='train_cnbc_data.pkl',
                 test_data_filename='test_cnbc_data.pkl',
                 tokenizer_filename='cnbc_tokenizers.pkl',
                 val_split=0.1,
                 embedding_size=25):
        self.val_split = val_split
        self.load_train_data(saved_dir, train_data_filename)
        self.load_test_data(saved_dir, test_data_filename)
        self.load_tokenizers(saved_dir, tokenizer_filename)
        self.embedding_size = embedding_size
        self.load_embeddings()

    def load_train_data(self, saved_dir, data_filename):
        data_pickle_path = os.path.join(saved_dir, data_filename)
        print('Loading', data_pickle_path, '...', end='')
        with open(data_pickle_path, 'rb') as f:
            td, ts = pickle.load(f)
            split_idx = self.calc_val_idx(len(td), len(ts))
            self.val_documents = td[:split_idx]
            self.val_summaries = ts[:split_idx]
            self.train_documents = td[split_idx:]
            self.train_summaries = ts[split_idx:]
        print('done.')

    def load_test_data(self, saved_dir, data_filename):
        data_pickle_path = os.path.join(saved_dir, data_filename)
        print('Loading', data_pickle_path, '...', end='')
        with open(data_pickle_path, 'rb') as f:
            (self.test_documents, self.test_summaries) = pickle.load(f)
        print('done.')

    def load_tokenizers(self, saved_dir, tokenizer_filename):
        tokenizer_pickle_path = os.path.join(saved_dir, tokenizer_filename)
        print('Loading', tokenizer_pickle_path, '...', end='')
        with open(tokenizer_pickle_path, 'rb') as f:
            (self.document_tokenizer, self.summary_tokenizer) = pickle.load(f)
        print('done.')

    def load_embeddings(self):
        print('Loading embeddings...', end='')
        self.embeddings = api.load('glove-twitter-' + str(self.embedding_size))
        print('done.')

    def calc_val_idx(self, train_len, test_len):
        total = train_len + test_len
        return floor(total * self.val_split)

    def index_to_vec(self, x):
        word = None
        try:
            word = self.document_tokenizer.index_word[x]
        except KeyError:
            if x == 0:
                word = 'pad'
            elif x == 41:
                word = 'start'
            elif x == 42:
                word = 'eos'
            else:
                word = 'unk'
        try:
            return self.embeddings[word]
        except KeyError:
            return np.zeros(shape=(self.embedding_size, ))

    def generator(self, batch_size=32, gen_type='train'):
        cur_i = 0
        docs = getattr(self, gen_type + '_documents')
        summaries = getattr(self, gen_type + '_summaries')

        while True:
            encoder_in = np.zeros(
                (batch_size, len(docs[0]), self.embedding_size),
                dtype='float32')
            decoder_target = np.zeros(
                (batch_size, len(summaries[0]), self.embedding_size),
                dtype='float32')
            for i, (input_text, target_text) in enumerate(
                    zip(docs[cur_i:cur_i + batch_size, :],
                        summaries[cur_i:cur_i + batch_size, :])):
                encoder_in[i] = np.array([self.index_to_vec(x) for x in input_text])
                decoder_target[i] = np.array([self.index_to_vec(x) for x in target_text])
            cur_i += batch_size
            yield encoder_in, decoder_target
