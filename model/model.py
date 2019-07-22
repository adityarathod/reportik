from tensorflow import keras
import tensorflow as tf
import numpy as np
from loader.dataman import DataManager


class NewsSummarizationModel:
    data = None
    encoder_shape = None
    decoder_shape = None
    model = None
    fixed_vector_dim = 200
    embedding_dim = 128

    def __init__(self, manager: DataManager):
        self.data = manager
        self.encoder_shape = {
            'n': self.data.documents.shape[1],
            'm': self.data.documents.shape[0],
        }
        self.decoder_shape = {
            'n': self.data.summaries.shape[1],
            'm': self.data.summaries.shape[0],
        }

    def build_model(self):
        # build encoder
        enc_in = keras.layers.Input(shape=(None,), name="Encoder_Input")
        enc_embedding = keras.layers.Embedding(self.data.document_vocab_size, self.embedding_dim, name="Input_Embedding")(enc_in)
        enc_lstm = keras.layers.LSTM(self.fixed_vector_dim, name="Encoder_LSTM", return_sequences=True, return_state=True)
        enc_out, h, c = enc_lstm(enc_embedding)
        enc_state = [h, c]

        # build decoder
        dec_in = keras.layers.Input(shape=(None,), name="Decoder_Input")
        dec_embedding = keras.layers.Embedding(self.data.summary_vocab_size, self.embedding_dim, name="Decoder_Embedding")(dec_in)
        dec_lstm = keras.layers.LSTM(self.fixed_vector_dim, name="Decoder_LSTM", return_sequences=True, return_state=True)
        dec_lstm_out, _, _ = dec_lstm(dec_in, initial_state=enc_state)
        dec_dense = keras.layers.Dense(self.decoder_shape['n'], name="Decoder_Dense", activation='softmax')
        dec_out = dec_dense(dec_lstm_out)

        # create model
        self.model = keras.Model([enc_in, dec_in], dec_out)
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

    def train(self):
        summary = np.array(self.data.summaries)
        decoder_input = summary[:, 1:]
        self.model.fit([self.data.documents, decoder_input], self.data.summaries)

    def view_document_text(self):
        # TODO: translate a sequence of numerical document tokens to actual text
        pass

    def view_summary_text(self):
        # TODO: translate a sequence of numerical summary tokens to actual text
        pass

if __name__ == '__main__':
    data = DataManager()
    model = NewsSummarizationModel(data)
    model.build_model()
    # tf.keras.utils.plot_model(model.model, to_file='model.png', show_shapes=True, show_layer_names=True)
    model.model.summary()
    print('training...')
    model.train()
    # model.model.save('model.h5')