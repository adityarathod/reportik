from tensorflow import keras
import numpy as np
import os
import pickle


class DataManager:
    documents = None
    document_tokenizer: keras.preprocessing.text.Tokenizer = None
    summaries = None
    summary_tokenizer: keras.preprocessing.text.Tokenizer = None

    def __init__(self, saved_dir='../data', data_filename='cnbc_data.pkl', tokenizer_filename='cnbc_tokenizers.pkl'):
        self.load_data(saved_dir, data_filename)
        self.load_tokenizers(saved_dir, tokenizer_filename)

    def load_data(self, saved_dir, data_filename):
        data_pickle_path = os.path.join(saved_dir, data_filename)
        print('Loading', data_pickle_path, '...', end='')
        with open(data_pickle_path, 'rb') as f:
            (self.documents, self.summaries) = pickle.load(f)
        print('done.')

    def load_tokenizers(self, saved_dir, tokenizer_filename):
        tokenizer_pickle_path = os.path.join(saved_dir, tokenizer_filename)
        print('Loading', tokenizer_pickle_path, '...', end='')
        with open(tokenizer_pickle_path, 'rb') as f:
            (self.document_tokenizer, self.summary_tokenizer) = pickle.load(f)
        print('done.')

    def training_generator(self, batch_size=32):
        cur_i = 0
        while True:
            encoder_in = np.zeros(
                (batch_size, len(self.documents[0])),
                dtype='int32')
            decoder_in = np.zeros(
                (batch_size, len(self.summaries[0])),
                dtype='int32')
            decoder_target = np.zeros(
                (batch_size, len(self.summaries[0]), self.summary_tokenizer.num_words), dtype='int32')
            for i, (input_text, target_text) in enumerate(
                    zip(self.documents[cur_i:cur_i + batch_size, :], self.summaries[cur_i:cur_i + batch_size, :])):
                for t, word in enumerate(input_text):
                    encoder_in[i, t] = word
                for t, word in enumerate(target_text):
                    decoder_in[i, t] = word
                    if t > 0:
                        decoder_target[i, t - 1, word] = 1.
            cur_i += batch_size
            yield [encoder_in, decoder_in], decoder_target
