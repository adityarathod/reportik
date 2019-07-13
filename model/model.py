from tensorflow import keras
import os
import pickle


class NewsSummarizationModel:
    summaries = None
    summaries_words = None
    summaries_words_reverse = None
    summary_vocab_size = None
    documents = None
    documents_words = None
    documents_words_reverse = None
    document_vocab_size = None
    encoder = None
    decoder = None
    fixed_vector_dim = 200
    embedding_dim = 128

    def __init__(self, pickled_data_path: str = '../data/'):
        # TODO: offload data loading to different functions to be called within the __init__ function
        pickle_path = os.path.join(pickled_data_path, 'cnbc_data.pkl')
        print(f'Opening {pickle_path}...')
        with open(pickle_path, 'rb') as fi:
            (self.documents, self.documents_words, self.documents_words_reverse),\
            (self.summaries, self.summaries_words, self.summaries_words_reverse) = pickle.load(fi)
        self.encoder = {
            'n': len(self.documents[0]),
            'm': len(self.documents),
            'input': None,
            'lstm': None,
            'states': None
        }
        self.decoder = {
            'n': len(self.summaries[0]),
            'm': len(self.summaries),
            'input': None,
            'lstm': None,
            'states': None
        }
        self.document_vocab_size = len(self.documents_words)
        self.summary_vocab_size = len(self.summaries_words)

    def build_encoder(self):
        self.encoder['input'] = keras.layers.Input(shape=(None, self.encoder['n']))
        self.encoder['lstm'] = keras.layers.LSTM(self.fixed_vector_dim, return_sequences=True, return_state=True)
        enc_out, h, c = self.encoder['lstm'](self.encoder['input'])
        self.encoder['states'] = [h, c]

    def build_decoder(self):
        # TODO: implement decoder
        pass

    def build_model(self):
        self.build_encoder()
        # TODO: call self.build_decoder() once written

    def view_document_text(self):
        # TODO: translate a sequence of numerical document tokens to actual text
        pass

    def view_summary_text(self):
        # TODO: translate a sequence of numerical summary tokens to actual text
        pass

if __name__ == '__main__':
    model = NewsSummarizationModel()
    model.build_encoder()