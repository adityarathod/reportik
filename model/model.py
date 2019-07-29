import os
import re

import numpy as np
import tensorflow as tf
from tensorflow import keras

from loader import DataManager
import seq2seq
import utils


class NewsSummarizationModel:
    model = None
    batch_size = None
    data = None
    text_replacements = {
        'U. S.': 'US',
        'U. K.': 'UK',
        'Sen.': 'Sen'
    }

    def __init__(self, manager: DataManager, batch_size=32):
        self.data = manager
        self.batch_size = batch_size

    def build_model(self, latent_dim=20):
        self.model = seq2seq.AttentionSeq2Seq(
            input_dim=25,
            input_length=len(self.data.train_documents[0]),
            hidden_dim=latent_dim,
            output_length=len(self.data.train_summaries[0]),
            output_dim=25,
            depth=4
        )

        self.model.compile(
            loss='mse',
            optimizer='rmsprop',
            metrics=['acc']
        )

    def train(self, epochs=1):
        cb = keras.callbacks.TensorBoard()
        self.model.fit_generator(
            self.data.generator(self.batch_size),
            epochs=epochs,
            steps_per_epoch=len(self.data.train_documents) // self.batch_size,
            validation_data=self.data.generator(self.batch_size, gen_type='val'),
            validation_steps=len(self.data.val_documents) // self.batch_size,
            callbacks=[cb]
        )

    def plot_model(self, image_path='model.png'):
        tf.keras.utils.plot_model(self.model, to_file=image_path, show_shapes=True, show_layer_names=True)

    def evaluate(self):
        return self.model.evaluate_generator(
            self.data.generator(self.batch_size, gen_type='test'),
            steps=len(self.data.test_documents) // self.batch_size
        )

    def save(self, path, filename='model'):
        self.model.save_weights(os.path.join(path, filename + '-seq2seq-attn-weights.h5'))

    def view_document_text(self, document):
        return self.data.document_tokenizer.sequences_to_texts([document])[0]

    def view_summary_text(self, summary):
        return self.data.summary_tokenizer.sequences_to_texts([summary])[0]

    def load(self, path):
        print('Loading saved model weights...')
        self.model.load_weights(path)

    def infer(self, document_text):
        max_doc_len = len(self.data.train_documents[0])
        doc_seq = self.data.document_tokenizer.texts_to_sequences([utils.clean_text(document_text)])
        doc_seq = keras.preprocessing.sequence.pad_sequences(doc_seq, max_doc_len, truncating='post')
        doc_seq = np.squeeze(doc_seq)
        doc_seq = np.array([self.data.index_to_vec(x) for x in doc_seq])
        doc_seq = np.reshape(doc_seq, (1, -1, self.data.embedding_size))
        summ_seq = self.model.predict(doc_seq)
        summ_seq = np.reshape(summ_seq, (150, self.data.embedding_size))
        print(summ_seq.shape)
        words = []
        for x in summ_seq:
            words.append(np.squeeze(self.data.embeddings.similar_by_vector(x, topn=1))[0])
        return ' '.join(words)


if __name__ == '__main__':
    data = DataManager()
    model = NewsSummarizationModel(data)
    model.build_model()
    model.model.summary()
    model.plot_model()
    print('training...')
    model.train()
