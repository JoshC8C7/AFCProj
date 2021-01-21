import requests
from newspaper import Article, Config, ArticleException
from newspaper.utils import BeautifulSoup
import pickle
import os

API_KEY = 'AIzaSyCpq7_EUObEz3azL3CrkZwK7OUIASMqLsA'
SEARCH_ENGINE_ID = 'be06938b6f07a2eb1'
CACHE_FILE='data/SearchCache.pickle'
SELECTED_SEARCH='google'

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'rb') as cache:
        web_cache = pickle.load(cache)
        print("Web Cache loaded")

def googleQ(term):
    print("searching ",term)
    num = 5
    url = f"https://www.googleapis.com/customsearch/v1?key={API_KEY}&cx={SEARCH_ENGINE_ID}&q={term}&num={num}"
    data = requests.get(url).json()
    found = data.get("items")
    return(i['link'] for i in found)

def bingFree(term):

    return ''

def bingS1(term):
    return ''

search_opts = {'google':googleQ, 'bingFree':bingFree,'bingS1':bingS1}



def searchFetch(term):
    if term in web_cache:
        return web_cache[term]
    else:
        searchFunct = search_opts[SELECTED_SEARCH]
        res = searchFunct(term)
        web_cache[term] = res
        return res


def nlpFeed(t):
    sources = set()
    config = Config()
    config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36'
    urls = googleQ(t)
    for url in urls:
        print("////////////// "+url + " ///////////")
        try:
            if ('www.bbc' in url):
                article = requests.get(url)
                if article.status_code != 200:
                    raise ArticleException
                soup = BeautifulSoup(article.content, 'html.parser')
                body = ' '.join(x.text for x in soup.findAll('p'))
                for a in body.split('. '):
                    sources.add(a)

            else:
                article = Article(url,config=config)
                article.download()
                article.parse()
                for a in article.text.split('. '):
                    sources.add(a)
        except ArticleException:
            print("Couldn't fetch: ", url)

    return sources

def dumpToCache():
    with open(CACHE_FILE, 'wb') as cache:
        pickle.dump(web_cache,cache)

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