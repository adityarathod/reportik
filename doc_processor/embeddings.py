import pickle
import fasttext
import numpy as np

embeddings_file = '../data/wiki-news-300d-1M.vec'

def load_data(file='../data/cnbc_data.pkl'):
    with open(file, 'rb') as fi:
        return pickle.load(fi)

def lookup_document_word(word_idx):
    return documents_words_reverse[word_idx]


(documents, documents_words, documents_words_reverse), \
(summaries, summaries_words, summaries_words_reverse) = load_data()

document_model = fasttext.load_model('../data/document_embeddings.bin')

docs_trans = np.zeros(shape=(len(documents), len(documents[0]), 100))

for idx, doc in enumerate(documents):
    print(f'processing document {idx}...')
    v = np.array([document_model.get_word_vector(lookup_document_word(x)) for x in doc])
    docs_trans[idx] = v

print(docs_trans.shape)