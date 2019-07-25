from loader import DataManager
from model.model import NewsSummarizationModel
import os


def main():
    manager = DataManager(saved_dir='./data')
    model = NewsSummarizationModel(manager)
    model.build_model()
    # model.model.summary()
    # model.plot_model()
    print('training...')
    model.train(epochs=2)
    model.save(os.getcwd(), 'cnbc')
    # model.load('./trained_model/cnbc-overall.h5', './trained_model/cnbc-encoder.h5', './trained_model/cnbc-decoder.h5')
    print(model.evaluate())
    print(model.infer('amazon is a piece of crap <PUNCT>'))

if __name__ == '__main__':
    main()