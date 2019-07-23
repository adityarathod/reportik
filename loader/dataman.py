from tensorflow import keras
from math import floor
import numpy as np
import os
import pickle


class DataManager:
    train_documents = None
    test_documents = None
    val_documents = None
    document_tokenizer: keras.preprocessing.text.Tokenizer = None
    train_summaries = None
    test_summaries = None
    val_summaries = None
    summary_tokenizer: keras.preprocessing.text.Tokenizer = None
    val_split = None

    def __init__(self, saved_dir='../data', train_data_filename='train_cnbc_data.pkl',
                 test_data_filename='test_cnbc_data.pkl',
                 tokenizer_filename='cnbc_tokenizers.pkl', val_split=0.1):
        self.val_split = val_split
        self.load_train_data(saved_dir, train_data_filename)
        self.load_test_data(saved_dir, test_data_filename)
        self.load_tokenizers(saved_dir, tokenizer_filename)

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

    def calc_val_idx(self, train_len, test_len):
        total = train_len + test_len
        return floor(total * self.val_split)

    def training_generator(self, batch_size=32):
        cur_i = 0
        while True:
            encoder_in = np.zeros(
                (batch_size, len(self.train_documents[0])),
                dtype='int32')
            decoder_in = np.zeros(
                (batch_size, len(self.train_summaries[0])),
                dtype='int32')
            decoder_target = np.zeros(
                (batch_size, len(self.train_summaries[0]), self.summary_tokenizer.num_words), dtype='int32')
            for i, (input_text, target_text) in enumerate(
                    zip(self.train_documents[cur_i:cur_i + batch_size, :], self.train_summaries[cur_i:cur_i + batch_size, :])):
                for t, word in enumerate(input_text):
                    encoder_in[i, t] = word
                for t, word in enumerate(target_text):
                    decoder_in[i, t] = word
                    if t > 0:
                        decoder_target[i, t - 1, word] = 1.
            cur_i += batch_size
            yield [encoder_in, decoder_in], decoder_target

    def test_generator(self, batch_size=32):
        cur_i = 0
        while True:
            encoder_in = np.zeros(
                (batch_size, len(self.test_documents[0])),
                dtype='int32')
            decoder_in = np.zeros(
                (batch_size, len(self.test_summaries[0])),
                dtype='int32')
            decoder_target = np.zeros(
                (batch_size, len(self.test_summaries[0]), self.summary_tokenizer.num_words), dtype='int32')
            for i, (input_text, target_text) in enumerate(zip(
                    self.test_documents[cur_i:cur_i + batch_size, :], self.test_summaries[cur_i:cur_i + batch_size, :]
            )):
                for t, word in enumerate(input_text):
                    encoder_in[i, t] = word
                for t, word in enumerate(target_text):
                    decoder_in[i, t] = word
                    if t > 0:
                        decoder_target[i, t - 1, word] = 1.
            cur_i += batch_size
            yield [encoder_in, decoder_in], decoder_target

    def val_generator(self, batch_size=32):
        cur_i = 0
        while True:
            encoder_in = np.zeros(
                (batch_size, len(self.val_documents[0])),
                dtype='int32')
            decoder_in = np.zeros(
                (batch_size, len(self.val_summaries[0])),
                dtype='int32')
            decoder_target = np.zeros(
                (batch_size, len(self.val_summaries[0]), self.summary_tokenizer.num_words), dtype='int32')
            for i, (input_text, target_text) in enumerate(zip(
                    self.val_documents[cur_i:cur_i + batch_size, :], self.val_summaries[cur_i:cur_i + batch_size, :]
            )):
                for t, word in enumerate(input_text):
                    encoder_in[i, t] = word
                for t, word in enumerate(target_text):
                    decoder_in[i, t] = word
                    if t > 0:
                        decoder_target[i, t - 1, word] = 1.
            cur_i += batch_size
            yield [encoder_in, decoder_in], decoder_target
