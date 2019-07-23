from tensorflow import keras
import tensorflow as tf
from loader import DataManager


class NewsSummarizationModel:
    model = None
    embedding_dim = None
    batch_size = None
    data = None

    def __init__(self, manager: DataManager, embedding_dim=100, batch_size=32):
        self.data = manager
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size

    def build_model(self):
        encoder_inputs = keras.layers.Input(shape=(None,))
        en_x = keras.layers.Embedding(self.data.document_tokenizer.num_words, self.embedding_dim)(encoder_inputs)
        encoder = keras.layers.LSTM(4, return_state=True)
        encoder_outputs, state_h, state_c = encoder(en_x)
        encoder_states = [state_h, state_c]

        decoder_inputs = keras.layers.Input(shape=(None,))
        dex = keras.layers.Embedding(self.data.summary_tokenizer.num_words, self.embedding_dim)
        final_dex = dex(decoder_inputs)
        decoder_lstm = keras.layers.LSTM(4, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(final_dex,
                                             initial_state=encoder_states)
        decoder_dense = keras.layers.Dense(data.summary_tokenizer.num_words, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        self.model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

    def train(self, epochs=1):
        cb = keras.callbacks.TensorBoard()
        self.model.fit_generator(
            data.training_generator(self.batch_size),
            epochs=epochs,
            steps_per_epoch=len(data.train_documents) // self.batch_size,
            validation_data=data.val_generator(self.batch_size),
            validation_steps=len(data.val_documents) // self.batch_size,
            callbacks=[cb]
        )

    def plot_model(self, image_path='model.png'):
        tf.keras.utils.plot_model(self.model, to_file=image_path, show_shapes=True, show_layer_names=True)

    def evaluate(self):
        return self.model.evaluate_generator(
            data.test_generator(self.batch_size),
            steps=len(data.test_documents) // self.batch_size
        )

    def save(self, path):
        self.model.save(path)

    def view_document_text(self, document):
        return self.data.document_tokenizer.sequences_to_texts([document])[0]

    def view_summary_text(self, summary):
        return self.data.summary_tokenizer.sequences_to_texts([summary])[0]

    def load_model(self, path):
        self.model = keras.models.load_model(path)

    def infer(self, document_text):
        doc_seq = self.data.document_tokenizer.texts_to_sequences([document_text])
        summ_seq = self.model.predict(doc_seq)
        return self.view_summary_text(summ_seq)

if __name__ == '__main__':
    data = DataManager()
    model = NewsSummarizationModel(data)
    model.build_model()
    model.model.summary()
    model.plot_model()
    print('training...')
    model.train()
