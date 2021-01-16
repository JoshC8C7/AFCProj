import requests
from newspaper import Article, Config
from pprint import pprint
import json
from newspaper.utils import BeautifulSoup
import lxml
API_KEY = 'AIzaSyCpq7_EUObEz3azL3CrkZwK7OUIASMqLsA'
SEARCH_ENGINE_ID = 'be06938b6f07a2eb1'


USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:78.0) Gecko/20100101 Firefox/78.0'

config = Config()
config.browser_user_agent = USER_AGENT
config.request_timeout = 10
base_url = 'https://www.bbc.com/news/health-54500673'
article = Article(base_url, config=config)
article.download()
article.parse()
"""
article_meta_data = article.meta_data

soup = BeautifulSoup(article.html, 'html.parser')

url = 'https://www.bbc.co.uk/news/world-europe-49345912'
article = requests.get(url)
soup = BeautifulSoup(article.content, 'html.parser')

body = soup.findAll('p')
pprint(list(x.text for x in body))"""





def nlpFeed(t):
    sources = set()
    config = Config()
    config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36'
    urls = googleQ(t)
    for url in urls:
        print("////////////// "+url + " ///////////")
        if ('www.bbc' in url):
            article = requests.get(url)
            soup = BeautifulSoup(article.content, 'html.parser')
            body = ' '.join(x.text for x in soup.findAll('p'))
            sources.add(body)
        else:
            article = Article(url,config=config)
            article.download()
            article.parse()
            sources.add(article.text)
    return sources

def googleQ(term):
    num = 5
    url = f"https://www.googleapis.com/customsearch/v1?key={API_KEY}&cx={SEARCH_ENGINE_ID}&q={term}&num={num}"
    data = requests.get(url).json()
    found = data.get("items")
    return(i['link'] for i in found)

nlpFeed('Covid: \'Urgent\' aviation support plea over travel curbs')


"""
def create():

    import json
    import csv
    safe =[]
    with open('data/csources.json') as json_file:
        data = json.load(json_file)
        for pk, pv in data.items():
            if pv.get('r',"") in ("VH","H","MF"):
                print(pk, pv['r'])
                safe.append([pk])

    with open('data/wikiS.csv') as wiki_file:
        reader = csv.reader(wiki_file)
        inList = list(reader)[1:]
    print(inList)
    for y in inList:
        x=y[0]
        if x[-1] == '/':
            print(x[:-1])
            safe.append([x[:-1]])
        else:
            print(x)
            safe.append([x])

    print(safe)
    with open('trusted.csv','w+',newline='') as f:
        writer = csv.writer(f)
        writer.writerows(safe)
        """