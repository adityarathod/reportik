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
    model.train()
    model.save(os.getcwd(), 'cnbc')
    # model.load_model('/Users/aditya/Desktop/saved_model.h5')
    # print(model.evaluate())
    # model.infer('American trade negotiators will soon head to China for face-to-face talks as the world’s two largest economies try to strike a deal, sources told CNBC <PUNCT> The US officials will travel to China for discussions sometime between Friday — the start of a six-week congressional recess in Washington — and Thursday, August 1 <PUNCT> While the talks represent a critical next step after a truce reached between the countries’ leaders in June, a deal is not viewed as near <PUNCT> President Donald Trump has signaled that he’d be willing to relax restrictions on China’s Huawei in exchange for purchases of US agricultural products <PUNCT> Longer-term, US officials have suggested they could roll back the tariffs in exchange for Beijing making the deal legally binding — something it backtracked on in May <PUNCT> White House officials are now suggesting that getting China to agree to the latter could take several months at least, even though Trump remains inclined to ink an eventual deal, according to three people familiar with the matter <PUNCT> In the meantime, the administration could shift its focus to ratifying the United States-Mexico-Canada Agreement <PUNCT> Trump sees approving his replacement for the North American Free Trade Agreement as a major economic and political priority <PUNCT> US Trade Representative Robert Lighthizer is using the remaining days of the congressional session to meet with lawmakers about the deal <PUNCT> CNBC has reported that the White House is expected to send the implementing legislation to Capitol Hill in September, setting up a vote on the deal this fall <PUNCT> Investors have watched the China talks closely <PUNCT> A widening trade war between Washington and Beijing would risk more damage to American companies and the global economy <PUNCT>  The Trump administration has put tariffs on $250 billion in Chinese goods — and threatened to put duties on even more products <PUNCT> Beijing has slapped tariffs on $110 billion in American goods <PUNCT>  Both the US and China have taken steps to deescalate tension in recent days <PUNCT> Trump agreed to make “timely” decisions about whether to allow American tech companies to sell to blacklisted Chinese firm Huawei <PUNCT> Meanwhile, Chinese state media reported that China had taken steps to start following through on its promise to buy more US agricultural goods <PUNCT> Trump sees the step as important to reaching an agreement as American farmers take a hit from tariffs on crops <PUNCT>')

if __name__ == '__main__':
    main()