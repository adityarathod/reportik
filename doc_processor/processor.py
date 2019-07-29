import os
import pickle

from sklearn.model_selection import train_test_split
from tensorflow import keras

import utils


class DocumentCleaner:
    texts = None
    texts_dir = None
    summaries = None
    summaries_dir = None
    texts_word_dict = None
    texts_rev_word_dict = None
    text_vocab_size = None
    summaries_word_dict = None
    summaries_rev_word_dict = None
    summaries_vocab_size = None
    text_tokenizer = None
    summary_tokenizer = None

    def __init__(self, texts='../data/texts', summaries='../data/points', text_replacements=None):
        self.texts_dir = texts
        self.summaries_dir = summaries
        self.texts = []
        self.summaries = []
        if text_replacements is not None:
            utils.text_cleaner.text_replacements = text_replacements

    def clean_texts(self):
        print(f'Ingesting and cleaning documents in {self.texts_dir}/...', end='')
        for file in os.listdir(self.texts_dir):
            if utils.is_hidden_file(file):
                continue
            with open(f'{self.texts_dir}/{file}', 'r', errors='ignore') as f:
                txt = f.read()
                self.texts.append(utils.clean_text(txt))
        print('done.')

    def tokenize_texts(self, max_length=1300, vocab_size=50000):
        print(f'Tokenizing documents...', end='')
        self.text_vocab_size = vocab_size
        tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token='<unk>',
                                                       filters='!"#$%&()*+,-–—./:;=?@[\\]^_`{|}~\t\n')
        tokenizer.fit_on_texts(self.texts)
        seq = tokenizer.texts_to_sequences(self.texts)
        self.texts = keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_length, truncating='post')
        word_dict = tokenizer.word_index
        word_dict['<pad>'] = 0
        tokenizer.index_word[0] = '<pad>'
        self.texts_word_dict = word_dict
        self.texts_rev_word_dict = dict([(value, key) for (key, value) in self.texts_word_dict.items()])
        self.text_tokenizer = tokenizer
        print('done.')

    def clean_summaries(self):
        print(f'Ingesting and cleaning summaries in {self.summaries_dir}/...', end='')
        for file in os.listdir(self.summaries_dir):
            if utils.is_hidden_file(file):
                continue
            with open(f'{self.summaries_dir}/{file}', 'r', errors='ignore') as f:
                txt = f.read()
                self.summaries.append(utils.clean_text(txt))
        print('done.')

    def tokenize_summaries(self, max_length=150, vocab_size=50000):
        print(f'Tokenizing summaries...', end='')
        self.summaries_vocab_size = vocab_size
        tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token='<unk>',
                                                       filters='!"#$%&()*+,-–—./:;=?@[\\]^_`{|}~\t\n')
        tokenizer.fit_on_texts(self.summaries)
        seq = tokenizer.texts_to_sequences(self.summaries)
        self.summaries = keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_length, truncating='post')
        word_dict = tokenizer.word_index
        word_dict['<pad>'] = 0
        tokenizer.index_word[0] = '<pad>'
        self.summaries_word_dict = word_dict
        self.summaries_rev_word_dict = dict([(value, key) for (key, value) in self.summaries_word_dict.items()])
        self.summary_tokenizer = tokenizer
        print('done.')

    def process(self):
        self.clean_texts()
        self.tokenize_texts()
        self.clean_summaries()
        self.tokenize_summaries()

    def split_and_dump_data(self, save_dir='../data', data_filename='cnbc_data.pkl',
                            tokenizer_filename='cnbc_tokenizers.pkl'):
        texts_train, texts_test, summ_train, summ_test = train_test_split(
            self.texts,
            self.summaries,
            test_size=0.2,
            random_state=420
        )
        with open(os.path.join(save_dir, 'train_' + data_filename), 'wb') as of:
            pickle.dump((texts_train, summ_train), of)
        with open(os.path.join(save_dir, 'test_' + data_filename), 'wb') as of:
            pickle.dump((texts_test, summ_test), of)
        with open(os.path.join(save_dir, tokenizer_filename), 'wb') as of2:
            pickle.dump((self.text_tokenizer, self.summary_tokenizer), of2)


if __name__ == '__main__':
    cleaner = DocumentCleaner(texts='../data/documents', summaries='../data/summaries')
    cleaner.process()
    cleaner.split_and_dump_data()
