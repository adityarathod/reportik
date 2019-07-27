from doc_processor.processor import DocumentCleaner
import fasttext
import os


def learn_embeddings(texts_dir='../data/documents', summaries_dir='../data/summaries'):
    print('Processing files located in', texts_dir, '...')
    document_file = open('./tmp-doc-emb.txt', 'w+')
    for file in os.listdir(texts_dir):
        if DocumentCleaner.is_hidden_file(file):
            continue
        with open(os.path.join(texts_dir, file), 'r', errors='ignore') as f:
            document_file.write(f.read() + '\n\n\n')
    print('Done.')
    document_file.close()
    print('Training word embeddings...')
    model = fasttext.train_unsupervised('./tmp-doc-emb.txt')
    model.save_model('doc_emb.bin')
    print('Done.')
    print('Processing files located in', summaries_dir, '...')
    summ_file = open('./tmp-summ-emb.txt', 'w+')
    for file in os.listdir(summaries_dir):
        if DocumentCleaner.is_hidden_file(file):
            continue
        with open(os.path.join(summaries_dir, file), 'r', errors='ignore') as f:
            summ_file.write(f.read() + '\n\n\n')
    print('Done.')
    summ_file.close()
    print('Training word embeddings...')
    model = fasttext.train_unsupervised('./tmp-summ-emb.txt')
    model.save_model('summ_emb.bin')
    print('Done.')

if __name__ == '__main__':
    learn_embeddings()