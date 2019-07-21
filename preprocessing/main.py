import tensorflow as tf
import tensorflow_datasets as tfds
from collections import Counter

def process_line(l: tf.Tensor):
    arr = tf.strings.split(l, sep=('\t' * 5))
    return arr[0], arr[1]


def load(data_path='../data/dataset.txt'):
    dataset = tf.data.TextLineDataset([data_path])
    paired_dataset = dataset.map(process_line)
    return paired_dataset

def create(data: tf.data.Dataset):
    tokenizer = tfds.features.text.Tokenizer()
    counter = Counter()
    # vocab = set()
    for doc, summ in data:
        doc_tokens = tokenizer.tokenize(doc.numpy())
        summ_tokens = tokenizer.tokenize(summ.numpy())
        counter.update(doc_tokens)
        counter.update(summ_tokens)
    return counter


if __name__ == '__main__':
    data = load()
    vocabulary = create(data)
    print(vocabulary.most_common(5))