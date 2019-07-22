from tensorflow import keras
import os
import pickle

class DataManager:
    documents = None
    document_tokenizer = None
    summaries = None
    summary_tokenizer = None

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