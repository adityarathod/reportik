import requests
import time
import newspaper
import hashlib
from bs4 import BeautifulSoup


class CNBCScraper:
    links = None
    logging = None
    sitemap_url = 'https://www.cnbc.com/CNBCsitemapAll9.xml'

    def __init__(self, sitemap_path: str = None, logging: bool = True):
        self.logging = logging
        if sitemap_path is not None:
            with open(sitemap_path) as f:
                sitemap = f.read()
                soup = BeautifulSoup(sitemap, 'xml')
                self.links = [x.string for x in soup.find_all('loc')]
        else:
            sitemap = requests.get(self.sitemap_url)
            soup = BeautifulSoup(sitemap.text(), 'xml')
            self.links = [x.string for x in soup.find_all('loc')]

    @staticmethod
    def generate_doc_id(title: str):
        md5 = hashlib.md5(f'{title}{str(int(time.time()))}'.encode())
        return md5.hexdigest()

    def log(self, msg, end='\n'):
        if self.logging:
            with open('logs.txt', 'a') as l:
                l.write(f'{msg}{end}')
            print(msg, end=end)

    def scrape(self):
        for idx, link in enumerate(self.links):
            if 'reuters-america-update-' in link:
                self.log(f'Skipping article #{idx + 1}...{link}')
                continue
            tmp = requests.get(link)
            soup = BeautifulSoup(tmp.text, 'html.parser')
            points = soup.find('div', class_='KeyPoints-list')
            if points != None:
                self.log(f'Downloading article #{idx + 1}...', end='')
                article = newspaper.Article(link)
                article.download()
                article.parse()
                key_pts = points.text
                article_id = CNBCScraper.generate_doc_id(article.title)
                with open(f'../data/texts/{article_id}.txt', 'w+') as f:
                    f.write(article.text)
                with open(f'../data/points/{article_id}.txt', 'w+') as f:
                    f.write(key_pts)
                self.log(f'"{article.title}"...', end='')
                self.log('done.')
            else:
                self.log(f'Skipping article #{idx + 1}...{link}')


if __name__ == '__main__':
    scraper = CNBCScraper(sitemap_path='cnbc_sitemap_r1.xml')
    scraper.scrape()