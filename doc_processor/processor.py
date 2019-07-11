from tensorflow import keras
from nltk.tokenize import sent_tokenize
import os
import re
import pickle

text_replacements = {
    'U. S.': 'US',
    'U. K.': 'UK'
}


class DocumentCleaner:
    texts = None
    texts_dir = None
    summaries_dir = None
    text_replacements = {
        'U. S.': 'US',
        'U. K.': 'UK'
    }

    def __init__(self, texts='../data/texts', summaries='../data/points', text_replacements=None):
        self.texts_dir = texts
        self.summaries_dir = summaries
        self.texts = []
        if text_replacements is not None:
            self.text_replacements = text_replacements

    @staticmethod
    def clean_sentence(s: str):
        out = s[:-1]
        for orig, new in text_replacements.items():
            out = out.replace(orig, new)
        return out + ' <PUNCT>'

    def clean_texts(self):
        for file in os.listdir(self.texts_dir):
            print(f'Processing {self.texts_dir}/{file}...')
            with open(f'{self.texts_dir}/{file}', 'r', errors='ignore') as f:
                txt = f.read()
                # summary punctuation fix
                fixed_txt = re.sub(r'\.(?=[^ \W\d])', '. ', txt)
                fixed_txt = fixed_txt.replace('\n', ' <NEWLINE> ')
                sentences = sent_tokenize(fixed_txt)
                sentences = [self.clean_sentence(s.strip()) for s in sentences]
                self.texts.append(' '.join(sentences) + '<EOS>')

    def dump_texts(self, path='../data/texts_clean.pkl'):
        with open(path, 'wb') as of:
            pickle.dump(self.texts, of)


if __name__ == '__main__':
    cleaner = DocumentCleaner()
    cleaner.clean_texts()
    cleaner.dump_texts()

#
# print(f'Tokenizing...')
# tokenizer = keras.preprocessing.text.Tokenizer(num_words=50000, oov_token='<UNK>')
# tokenizer.fit_on_texts(texts)
# seq = tokenizer.texts_to_sequences(texts)
# padded = keras.preprocessing.sequence.pad_sequences(seq)
# word_dict = tokenizer.word_index

# print(word_dict)
# word_vectors = api.load('glove-wiki-gigaword-100')