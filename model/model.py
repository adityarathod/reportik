from tensorflow import keras
from loader import DataManager
import tensorflow as tf
import numpy as np
import os


class NewsSummarizationModel:
    model = None
    encoder_model = None
    decoder_model = None
    embedding_dim = None
    batch_size = None
    data = None

    def __init__(self, manager: DataManager, embedding_dim=100, batch_size=32):
        self.data = manager
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size

    def build_model(self):
        latent_dim = 32
        embedding_dim = 200
        encoder_inputs = keras.Input(shape=(None,), name='encoder_inputs')
        encoder_embedding = keras.layers.Embedding(input_dim=self.data.document_tokenizer.num_words,
                                                   output_dim=embedding_dim,
                                                   input_length=len(self.data.train_documents[0]),
                                                   name='encoder_embedding')
        encoder_lstm = keras.layers.LSTM(units=latent_dim, return_state=True, name='encoder_lstm')
        encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_embedding(encoder_inputs))
        encoder_states = [encoder_state_h, encoder_state_c]

        decoder_inputs = keras.Input(shape=(None, self.data.summary_tokenizer.num_words), name='decoder_inputs')
        decoder_lstm = keras.layers.LSTM(units=latent_dim, return_state=True, return_sequences=True,
                                         name='decoder_lstm')
        decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = keras.layers.Dense(units=self.data.summary_tokenizer.num_words, activation='softmax',
                                           name='decoder_dense')
        decoder_outputs = decoder_dense(decoder_outputs)

        self.model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=keras.optimizers.RMSprop(lr=1e-4),
            metrics=['acc']
        )

        self.encoder_model = keras.Model(encoder_inputs, encoder_states)

        decoder_state_inputs = [keras.Input(shape=(latent_dim,)), keras.Input(shape=(latent_dim,))]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_state_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = keras.Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)

    def train(self, epochs=1):
        cb = keras.callbacks.TensorBoard()
        self.model.fit_generator(
            self.data.training_generator(self.batch_size),
            epochs=epochs,
            steps_per_epoch=len(self.data.train_documents) // self.batch_size,
            validation_data=self.data.val_generator(self.batch_size),
            validation_steps=len(self.data.val_documents) // self.batch_size,
            callbacks=[cb]
        )

    def plot_model(self, image_path='model.png'):
        tf.keras.utils.plot_model(self.model, to_file=image_path, show_shapes=True, show_layer_names=True)

    def evaluate(self):
        return self.model.evaluate_generator(
            self.data.test_generator(self.batch_size),
            steps=len(self.data.test_documents) // self.batch_size
        )

    def save(self, path, filename='model'):
        self.model.save(os.path.join(path, filename + '-overall.h5'))
        self.encoder_model.save(os.path.join(path, filename + '-encoder.h5'))
        self.decoder_model.save(os.path.join(path, filename + '-decoder.h5'))

    def view_document_text(self, document):
        return self.data.document_tokenizer.sequences_to_texts([document])[0]

    def view_summary_text(self, summary):
        return self.data.summary_tokenizer.sequences_to_texts([summary])[0]

    def load(self, overall_model_path, encoder_path, decoder_path):
        print('Loading saved models...')
        self.model = keras.models.load_model(overall_model_path)
        self.encoder_model = keras.models.load_model(encoder_path, compile=False)
        self.decoder_model = keras.models.load_model(decoder_path, compile=False)

    def _gl(self, name):
        return self.model.get_layer(name)

    def infer(self, document_text):
        max_seq_len = len(self.data.train_summaries[0])

        doc_seq = self.data.document_tokenizer.texts_to_sequences([document_text + ' <EOS>'])
        doc_seq = keras.preprocessing.sequence.pad_sequences(doc_seq, max_seq_len, truncating='post')

        tar_seq = np.zeros((1, 1, self.data.summary_tokenizer.num_words))
        tar_seq[0, 0, self.data.summary_tokenizer.word_index['<start>']] = 1.

        states_value = self.encoder_model.predict(doc_seq)

        stop = False
        summ_seq = []

        while not stop:
            out_tok, h, c = self.decoder_model.predict([tar_seq] + states_value)
            s_tok_idx = np.argmax(out_tok[0, -1, :])
            summ_seq.append(s_tok_idx)

            if s_tok_idx != 0:
                if self.data.summary_tokenizer.index_word[s_tok_idx] == '<eos>' or len(summ_seq) >= max_seq_len:
                    stop = True
            else:
                if len(summ_seq) >= max_seq_len:
                    stop = True

            target_seq = np.zeros((1, 1, self.data.summary_tokenizer.num_words))
            target_seq[0, 0, s_tok_idx] = 1.

            states_value = [h, c]

        return self.view_summary_text(summ_seq)


if __name__ == '__main__':
    data = DataManager()
    model = NewsSummarizationModel(data)
    model.build_model()
    model.model.summary()
    model.plot_model()
    print('training...')
    model.train()
