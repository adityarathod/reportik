import os

from loader import DataManager
from model.model import NewsSummarizationModel

example_text = '''In a week that brought earnings from the top online advertising companies, Google and Facebook showed that they still dominate the market. But smaller players are growing faster and picking up market share.

In addition to Facebook and Alphabet, which control a combined 51% of the global digital ad market, Snap, Amazon and Twitter all reported results this week. Snap recorded the highest growth rate of the group, expanding 48% from a year earlier. Revenue at Facebook climbed 28%, while Google’s ad business generated growth of 16%. EMarketer predicted earlier this year that Google and Facebook’s cumulative market share in the U.S. would drop in 2019.

Snap is still losing a lot of money ($255 million in the second quarter), but advertisers are clearly excited about its young devoted user base and are looking for ways to diversify their spending beyond the Facebook-Google duopoly. Snap shares surged 19% on Wednesday, finally jumping past the $17 IPO price from 2017.

“Notwithstanding fierce competition for user mindshare and advertiser dollars and a history of being hugely unprofitable, progress towards profitability, improving user growth trajectory, strong traction among advertisers, and sustained cost control have benefited Snap’s outlook,” wrote Michael Pachter, an analyst at Wedbush Securities, in a note following the company’s earnings report. He has a neutral rating on the stock.

Global digital ad spend is expected to reach $333.25 billion in 2019, according to eMarketer. Behind Google and Facebook comes China’s Alibaba and then Amazon. Rounding out the top six are two other Chinese companies, Baidu and Tencent.



Amazon doesn’t break out advertising, but it’s the biggest part of the company’s “Other” category, which grew 37% to $3 billion.

On Friday, Twitter posted better-than-expected results, with advertising revenue jumping 21% from a year earlier, thanks to 29% growth in the U.S. The stock jumped almost 9% on the report.

Chief Financial Officer Ned Segal said on the earnings call that “our messages of launching new products and services and connecting what’s happening continues to really resonate with advertisers.”

While Facebook and Alphabet also exceeded estimates — and Alphabet shares soared on its report — advertisers are clearly seeing opportunities on other platforms to reach different sets of eyeballs. Whatever pressure Facebook and Google are facing from lawmakers and regulators has yet to make its way into their top-line numbers, but potential limits on their growth could give competitors another edge from here.



Much of Snap’s momentum is coming from enhanced products that are luring consumers. The company’s lenses are popular ways for users to see what they’d look like as a person of the opposite sex or as a baby. It’s also given advertisers more opportunities to reach those users, including through “Snap Select,” which lets brands run non-skip commercials on the part of the Snapchat app that has shows.

“The popularity of these Lenses drew millions of people into our rebuilt Android application, where they experienced the new and improved Snapchat that led to increased engagement,” CEO Evan Spiegel said in his prepared remarks to investors. “The enhancements we have made to our advertising business and self-serve platform meant that we were better able to monetize this increased engagement, leading to accelerating revenue growth.”

EMarketer expects Snapchat to generate $1.36 billion in net worldwide ad revenue this year, a 30% increase over 2018, giving the company a 0.4% share of the worldwide digital ad market.

Amazon has built its lucrative ad business primarily by charging brands to promote their products in various ways on the shopping site and app and in streaming videos. On the earnings call, CFO Brian Olsavsky said the company is “adding more and more advertising as we roll out devices and Prime Video — new Prime Video content in particular internationally.”

Dave Fildes, Amazon’s director of investor relations, added that the company has taken steps with video advertising on live sports and IMDb TV. According to eMarketer, Amazon’s worldwide ad revenue is expected to reach $14.03 billion in 2019, giving it a 4.2% share of worldwide digital ad spend.

Twitter, meanwhile, is expected to record ad revenue of $2.97 billion in 2019, according to eMarketer, giving it 0.9% of the market. The site just got a major redesign that didn’t add the ability to edit tweets but instead focused on moving things like “trending topics” around and making it easier to join conversations and manage direct messages.

Not everyone loves the new look.

“Twitter’s value proposition to advertisers is not the size of its audience, but the engagement of its users,” eMarketer Senior Analyst Jasmine Enberg said in an emailed statement. “Next quarter’s earnings will show whether Twitter can keep up the growth momentum amidst negative user feedback over the website redesign rolled out in July.”'''


def main():
    manager = DataManager(saved_dir='./data')
    model = NewsSummarizationModel(manager)
    model.build_model()
    model.model.summary()
    # model.plot_model()
    print('training...')
    model.train(epochs=2)
    model.save(os.getcwd(), 'cnbc')
    # model.load('./trained_model/cnbc-overall.h5', './trained_model/cnbc-encoder.h5', './trained_model/cnbc-decoder.h5')
    print(model.evaluate())
    print(model.infer(example_text))


if __name__ == '__main__':
    main()
