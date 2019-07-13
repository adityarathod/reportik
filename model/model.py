from tensorflow import keras
import os
import pickle


class NewsSummarizationModel:
    summaries = None
    summaries_words = None
    summaries_words_reverse = None
    documents = None
    documents_words = None
    documents_words_reverse = None
    encoder = None
    decoder = None

    def __init__(self, pickled_data_path: str = '../data/'):
        # TODO: offload data loading to different functions to be called within the __init__ function
        pickle_path = os.path.join(pickled_data_path, 'cnbc_data.pkl')
        print(f'Opening {pickle_path}...')
        with open(pickle_path, 'rb') as fi:
            (self.documents, self.documents_words, self.documents_words_reverse),\
            (self.summaries, self.summaries_words, self.summaries_words_reverse) = pickle.load(fi)
            self.encoder['n'] = len(self.documents[0])
            self.encoder['m'] = len(self.documents)
            self.decoder['n'] = len(self.decoder[0])
            self.decoder['m'] = len(self.decoder)
        self.encoder = {
            'n': len(self.documents[0]),
            'm': len(self.documents),
            'layers': {}
        }
        self.decoder = {
            'n': len(self.summaries[0]),
            'm': len(self.summaries),
            'layers': {}
        }

    def build_encoder(self):
        # TODO: implement encoder layers as per spec
        self.encoder.layers = {
            'input': keras.layers.Input(shape=(None, self.encoder.n)),
            # 'embedding': keras.layers.Embedding()
        }

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
