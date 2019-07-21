import os
import pickle

class DataManager:
    summaries = None
    summaries_words = None
    summaries_words_reverse = None
    summary_vocab_size = None
    documents = None
    documents_words = None
    documents_words_reverse = None
    document_vocab_size = None

    def __init__(self, pickled_data_path: str = '../data/', filename: str='cnbc_data.pkl'):
        pickle_path = os.path.join(pickled_data_path, filename)
        print(f'Opening {pickle_path}...')
        with open(pickle_path, 'rb') as fi:
            (self.documents, self.documents_words, self.documents_words_reverse, self.document_vocab_size), \
            (self.summaries, self.summaries_words, self.summaries_words_reverse, self.summary_vocab_size) = pickle.load(fi)