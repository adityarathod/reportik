from tensorflow import keras
from nltk.tokenize import sent_tokenize
import os
import re
import pickle
import numpy as np

# Data can also be downloaded from https://dl.bintray.com/applecrazy/Reportik-CNBC-Data/cnbc_data.pkl


class DocumentCleaner:
    texts = None
    texts_dir = None
    summaries = None
    summaries_dir = None
    text_replacements = {
        'U. S.': 'US',
        'U. K.': 'UK',
        'Sen.': 'Sen'
    }
    texts_word_dict = None
    texts_rev_word_dict = None
    summaries_word_dict = None
    summaries_rev_word_dict = None

    def __init__(self, texts='../data/texts', summaries='../data/points', text_replacements=None):
        self.texts_dir = texts
        self.summaries_dir = summaries
        self.texts = []
        self.summaries = []
        if text_replacements is not None:
            self.text_replacements = text_replacements

    @staticmethod
    def is_hidden_file(name: str):
        return name.startswith('.')

    def clean_sentence(self, s: str):
        out = s[:-1]
        for orig, new in self.text_replacements.items():
            out = out.replace(orig, new)
        return out + ' <PUNCT> '

    def clean_texts(self):
        print(f'Processing {self.texts_dir}/...')
        for file in os.listdir(self.texts_dir):
            if self.is_hidden_file(file):
                continue
            with open(f'{self.texts_dir}/{file}', 'r', errors='ignore') as f:
                txt = f.read()
                # summary punctuation fix
                fixed_txt = re.sub(r'\.(?=[^ \W\d])', '. ', txt)
                fixed_txt = fixed_txt.replace('\n', ' <NEWLINE> ')
                sentences = sent_tokenize(fixed_txt)
                sentences = [self.clean_sentence(s.strip()) for s in sentences]
                self.texts.append('<START> ' + ''.join(sentences) + '<EOS>')

    def tokenize_texts(self):
        print(f'Tokenizing texts...')
        tokenizer = keras.preprocessing.text.Tokenizer(num_words=50000, oov_token='<unk>', filters='!"#$%&()*+,-–—./:;=?@[\\]^_`{|}~\t\n')
        tokenizer.fit_on_texts(self.texts)
        seq = tokenizer.texts_to_sequences(self.texts)
        self.texts = keras.preprocessing.sequence.pad_sequences(seq)
        word_dict = tokenizer.word_index
        word_dict['<pad>'] = 0
        self.texts_word_dict = word_dict
        self.texts_rev_word_dict = dict([(value, key) for (key, value) in self.texts_word_dict.items()])

    def decode_text_sequence(self, seq):
        return ' '.join([self.texts_rev_word_dict.get(i, '?') for i in seq])

    def decode_summary_sequence(self, seq):
        return ' '.join([self.summaries_rev_word_dict.get(i, '?') for i in seq])

    def dump_texts(self, save_path='../data'):
        with open(f'{save_path}/texts_clean.pkl', 'wb') as of:
            pickle.dump(self.texts, of)
        with open(f'{save_path}/text_word_dict.pkl', 'wb') as of:
            pickle.dump(self.texts_word_dict, of)
        with open(f'{save_path}/text_word_dict_rev.pkl', 'wb') as of:
            pickle.dump(self.texts_rev_word_dict, of)

    def clean_summaries(self):
        print(f'Processing {self.summaries_dir}/...')
        for file in os.listdir(self.summaries_dir):
            if self.is_hidden_file(file):
                continue
            with open(f'{self.summaries_dir}/{file}', 'r', errors='ignore') as f:
                txt = f.read()
                # summary punctuation fix
                fixed_txt = re.sub(r'\.(?=[^ \W\d])', '. ', txt)
                fixed_txt = fixed_txt.replace('\n', ' <NEWLINE> ')
                fixed_txt = fixed_txt.replace('\xa0', ' ')
                sentences = sent_tokenize(fixed_txt)
                sentences = [self.clean_sentence(s.strip()) for s in sentences]
                self.summaries.append('<START> ' + ''.join(sentences) + '<EOS>')

    def tokenize_summaries(self):
        print(f'Tokenizing summaries...')
        tokenizer = keras.preprocessing.text.Tokenizer(num_words=50000, oov_token='<unk>', filters='!"#$%&()*+,-–—./:;=?@[\\]^_`{|}~\t\n')
        tokenizer.fit_on_texts(self.summaries)
        seq = tokenizer.texts_to_sequences(self.summaries)
        self.summaries = keras.preprocessing.sequence.pad_sequences(seq)
        word_dict = tokenizer.word_index
        word_dict['<pad>'] = 0
        self.summaries_word_dict = word_dict
        self.summaries_rev_word_dict = dict([(value, key) for (key, value) in self.summaries_word_dict.items()])

    def dump_data(self, save_path='../data'):
        with open(f'{save_path}/cnbc_data.pkl', 'wb') as of:
            pickle.dump(
                (
                    (self.texts, self.texts_word_dict, self.texts_rev_word_dict),
                    (self.summaries, self.summaries_word_dict, self.summaries_rev_word_dict)
                ),
                of
            )

    def dump_summaries(self, save_path='../data'):
        with open(f'{save_path}/summaries_clean.pkl', 'wb') as of:
            pickle.dump(self.summaries, of)
        with open(f'{save_path}/summaries_word_dict.pkl', 'wb') as of:
            pickle.dump(self.summaries_word_dict, of)
        with open(f'{save_path}/summaries_word_dict_rev.pkl', 'wb') as of:
            pickle.dump(self.summaries_rev_word_dict, of)


if __name__ == '__main__':
    cleaner = DocumentCleaner()
    cleaner.clean_texts()
    cleaner.tokenize_texts()
    # cleaner.dump_texts()
    cleaner.clean_summaries()
    cleaner.tokenize_summaries()
    # cleaner.dump_summaries()
    cleaner.dump_data()
