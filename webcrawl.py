import requests
from newspaper import Article, Config, ArticleException
from newspaper.utils import BeautifulSoup
from azure.cognitiveservices.search.customsearch import CustomSearchClient
from msrest.authentication import CognitiveServicesCredentials
import pickle
import os
import time
import json
from pprint import pprint

GOOGLE_API_KEY = 'AIzaSyCpq7_EUObEz3azL3CrkZwK7OUIASMqLsA'
BING_FREE_KEY = 'e5da3b0fd59a4161ad4aac8b8c5535b0'
BING_S1_KEY = 'fb85fa996ee84161b15d2f79186cd405'

CACHE_FILE='data/SearchCache.pickle'
SELECTED_SEARCH='bingFree'

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'rb') as cache:
        web_cache = pickle.load(cache)
        print("Web Cache loaded")

def googleQ(term):
    print("searching ",term)
    num = 5
    url = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_API_KEY}&cx=be06938b6f07a2eb1&q={term}&num={num}"
    data = requests.get(url).json()
    found = data.get("items")
    return(i['link'] for i in found)

def bingFree(term):
    return bingParse(term, BING_FREE_KEY)

def bingS1(term):
    return bingParse(term, BING_S1_KEY)

def bingParse(term, key):
    time.sleep(1)
    url = f"https://api.bing.microsoft.com/v7.0/custom/search?q={term}&customconfig=c43aa9a7-40ee-4261-8ead-124b5a0ddcbc&mkt=en-GB"
    data = requests.get(url, headers={"Ocp-Apim-Subscription-key": key})
    vals = data.json().get('webPages',{}).get('value',{})
    if not vals:
        return []
    else:
        return list(i['url'] for i in vals)

search_opts = {'google':googleQ, 'bingFree':bingFree,'bingS1':bingS1}


def searchFetch(term):
    if term in web_cache:
        print("Cache Hit: ", term)
        return web_cache[term]
    else:
        print("Cache miss ", term)
        searchFunct = search_opts[SELECTED_SEARCH]
        res = searchFunct(term)
        print("Writing to cache -",res,"-")
        web_cache[term] = res
        return res


def nlpFeed(t):
    sources = set()
    config = Config()
    config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36'
    urls = searchFetch(t)
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
    dumpToCache()
    return sources

def dumpToCache():
    with open(CACHE_FILE, 'wb') as cache:
        pickle.dump(web_cache,cache)

if __name__ == "__main__":
    resp = input("Cache management. Run main.py for standard route. Enter 'clear' to clear URL cache, 'inspect' to view it, or 'exit'.")
    if resp == 'inspect':
        with open(CACHE_FILE, 'rb') as cache:
            print(pickle.load(cache))
    elif resp == 'clear':
        with open(CACHE_FILE, 'wb') as cache:
            pickle.dump({},cache)
        print("Cache cleared")


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