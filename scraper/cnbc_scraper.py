from bs4 import BeautifulSoup
from multiprocessing.dummy import Pool as ThreadPool
import requests
import time
import os
import newspaper
import re
import hashlib


class CNBCScraper:
    links = None
    logging = None
    documents_path = None
    summaries_path = None

    def __init__(self, sitemap_path, logging: bool = True):
        self.logging = logging
        with open(sitemap_path) as f:
            sitemap = f.read()
            soup = BeautifulSoup(sitemap, 'xml')
            self.links = [x.string for x in soup.find_all('loc')]

    @staticmethod
    def generate_doc_id(title: str):
        md5 = hashlib.md5(f'{title}'.encode())
        return md5.hexdigest()

    def log(self, msg, end='\n'):
        if self.logging:
            with open('logs.txt', 'a') as l:
                l.write(f'{msg}{end}')
            print(msg, end=end)

    def process_link(self, enum):
        idx, link = enum
        if 'reuters-america-update-' in link:
            self.log(f'Skipping article #{idx + 1}...{link}')
            return
        tmp = requests.get(link)
        soup = BeautifulSoup(tmp.text, 'html.parser')
        points = soup.find('div', class_='KeyPoints-list')
        if points is not None:
            article = newspaper.Article(link)
            article.download()
            article.parse()
            key_pts = points.text
            key_pts = re.sub(r'\.(?=[^ \W\d])', '. ', key_pts)
            article_id = self.generate_doc_id(article.title)
            with open(os.path.join(self.documents_path, str(article_id) + '.txt'), 'w+') as f:
                f.write(article.text)
            with open(os.path.join(self.summaries_path, str(article_id) + '.txt'), 'w+') as f:
                f.write(key_pts)
            self.log(f'Downloading article #{idx + 1}: "{article.title}..."')
        else:
            self.log(f'Skipping article #{idx + 1}...{link}')

    def scrape(self, documents_path, summaries_path):
        self.documents_path = documents_path
        self.summaries_path = summaries_path
        pool = ThreadPool(4)
        _ = pool.map(self.process_link, enumerate(self.links))


if __name__ == '__main__':
    scraper = CNBCScraper('cnbc_sitemap_2018.xml')
    doc_path = os.path.join('..', 'data', 'documents')
    sum_path = os.path.join('..', 'data', 'summaries')
    scraper.scrape(documents_path=doc_path, summaries_path=sum_path)