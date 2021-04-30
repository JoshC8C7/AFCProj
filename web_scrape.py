import requests
from newspaper import Article, Config, ArticleException
from newspaper.utils import BeautifulSoup

import pickle
import os

# Tokens and user-agent string are stored in tokens.py (omitted from Git for security).
import tokens

BING_KEY = tokens.BING_S1_KEY
GOOGLE_API_KEY = tokens.GOOGLE_API_KEY
AGENT = tokens.AGENT


# Import cache file containing searches and documents, specify which search engine/config to use.
CACHE_FILE_SEARCH = 'data/SearchingCache.pickle'
CACHE_FILE_ARTICLE = 'data/ArticleCache.pickle'
LIMITED_CACHE = 'data/LimitedCache.pickle'
SELECTED_SEARCH = tokens.SEARCH
EVIDENCE_BATCH_SIZE = 5
OPEN_SEARCH_CONFIG = '4f2142cb-2875-478f-b6a1-da7beabdec7b&mkt=en-GB&count=' + str(EVIDENCE_BATCH_SIZE) #AFC-Limited @ Bing
GOOGLE_PF_ONLY = 'e1d9abb2728495ee9'

if not os.path.exists(CACHE_FILE_SEARCH):
    with open(CACHE_FILE_SEARCH, 'wb') as cache:
        pickle.dump({}, cache)
with open(CACHE_FILE_SEARCH, 'rb') as cache:
    search_cache = pickle.load(cache)
    print("Web Cache loaded")

if not os.path.exists(CACHE_FILE_ARTICLE):
    with open(CACHE_FILE_ARTICLE, 'wb') as cache:
        pickle.dump({}, cache)
with open(CACHE_FILE_ARTICLE, 'rb') as cache:
    article_cache = pickle.load(cache)
    print("Article Cache loaded")

if not os.path.exists(LIMITED_CACHE):
    with open(LIMITED_CACHE, 'wb') as cache:
        pickle.dump({}, cache)
with open(LIMITED_CACHE, 'rb') as cache:
    limited_cache = pickle.load(cache)
    print("Limited Cache loaded")


# Define various Search functions:

# Fetch data from search limited to the first politifact result (for closed-domain evaluation). There was initially
# more pre-processing done before passing to searchParse; these functions remain for extensibility (e.g. with google).
def politifact_only(term):
    print("searching ",term)
    url = f"https://www.googleapis.com/customsearch/v1/siterestrict?key={GOOGLE_API_KEY}&cx={GOOGLE_PF_ONLY}&q={term}&num=1"
    data = requests.get(url).json()
    print(data)
    found = data.get("items")
    return list(i['link'] for i in found)


def bing_restricted_domain(term,config=OPEN_SEARCH_CONFIG):
    tk=term.replace(" ","%20")
    url = f"https://api.bing.microsoft.com/v7.0/custom/search?q={tk}&customconfig={config}"
    data_in = requests.get(url, headers={"Ocp-Apim-Subscription-key": BING_KEY})
    vals = data_in.json().get('webPages', {}).get('value', {})
    return [] if not vals else list(i['url'] for i in vals)


# Dictionary of available search options.
search_opts = {'bing': bing_restricted_domain, 'pfOnly': politifact_only}


# Trigger a search, first checking if the term has been sought from the cache.
def search_fetch(term,limiter):
    if limiter:
        print("Limited Search with param: ", limiter)
        if term in limited_cache:
            print("Cache Hit: ", term)
            return limited_cache[term]
        else:
            print("Cache miss ", term)
            res = search_opts[limiter](term)
            print("Writing to cache -", res, "-")
            limited_cache[term] = res
            return res

    if term in search_cache:
        print("Cache Hit: ", term)
        return search_cache[term]
    else:
        print("Cache miss ", term)
        search_funct = search_opts[SELECTED_SEARCH]
        res = search_funct(term)
        print("Writing to cache -", res, "-")
        search_cache[term] = res
        return res


# Takes a term and returns parsed article text
def nlp_feed(term,limiter=None):
    fresh_cache = True
    sources = set()
    config = Config()
    config.browser_user_agent = AGENT
    urls = search_fetch(term,limiter)
    print(urls)
    for url in urls[:1+EVIDENCE_BATCH_SIZE]:

        # Exclude any links (usually Associated Press ones) which are themselves fact-checks, which would be cheating...
        #unless running on closed domain, in which case 'factcheck' is in every domain...
        if limiter is None and 'factcheck' in url or 'fact-check' in url:
            continue

        # Scrape requested URLs if they aren't currently in cache.
        if url in article_cache:
            wc = article_cache[url]
            if wc != '':
                sources.add((url, wc))
                print("Cache Hit on ", url[:max(len(url), 50)], "......")
        else:
            try:
                # newspapers can't handle bbc ergo custom approach
                if 'www.bbc' in url:
                    article = requests.get(url)
                    if article.status_code != 200:
                        raise ArticleException
                    soup = BeautifulSoup(article.content, 'html.parser')
                    body = ' '.join(z.text for z in soup.findAll('p'))
                    sources.add((url, body))
                    if body not in article_cache:
                        article_cache[url] = body
                        fresh_cache = False

                else:
                    article = Article(url, config=config, language='en')
                    article.download()
                    article.parse()
                    sources.add((url, article.text))
                    if article.text not in article_cache:
                        article_cache[url] = article.text
                        fresh_cache = False

            except ArticleException:
                print("Couldn't fetch: ", url)
                article_cache[url] = ''
                fresh_cache = False

    # Only dump to disk if cache has been modified
    if not fresh_cache:
        dump_to_disk()
    return sources


# Dumps cache to disk. Note that dumping the cache does not entail reloading it, however assuming one user at a time
# them ephemeral cache will always be up to date with disk cache, once written here.
def dump_to_disk():
    with open(CACHE_FILE_ARTICLE, 'wb') as cache_file:
        pickle.dump(article_cache, cache_file)
    with open(CACHE_FILE_SEARCH, 'wb') as cache_file:
        pickle.dump(search_cache, cache_file)
    with open(LIMITED_CACHE, 'wb') as cache_file:
        pickle.dump(limited_cache, cache_file)


# Utility code for managing cache.
if __name__ == "__main__":
    import csv
    with open(CACHE_FILE_SEARCH, 'rb') as cache:
        kd = (pickle.load(cache))
        with open("sourcesOut.csv",'w') as out:
            writer = csv.writer(out)
            writer.writerows(list(kd.values()))



    resp = input(
        "Cache management. Enter 'clear' to clear URL cache, 'inspect' to view it, "
        "'create' to re-form the trusted source list, or 'exit'.")
    if resp == 'inspect':
        with open(CACHE_FILE_ARTICLE, 'rb') as cache:
            kd = (pickle.load(cache))
            print(kd.keys())
        with open(CACHE_FILE_SEARCH, 'rb') as cache:
            kd = (pickle.load(cache))
            print(kd.keys())
    elif resp == 'clear':
        with open(CACHE_FILE_ARTICLE, 'wb') as cache:
            pickle.dump({}, cache)
        print("Cache cleared")
        with open(CACHE_FILE_SEARCH, 'rb') as cache:
            kd = (pickle.load(cache))
            print(kd.keys())
    elif resp == 'create':
        import json
        import csv

        safe = []
        with open('data/sources/csources.json') as json_file:
            data = json.load(json_file)
            for pk, pv in data.items():
                if pv.get('r', "") in ("VH", "H", "MF"):
                    print(pk, pv['r'])
                    safe.append([pk])

        with open('data/sources/wikiS.csv') as wiki_file:
            reader = csv.reader(wiki_file)
            inList = list(reader)[1:]
        print(inList)
        for y in inList:
            x = y[0]
            if x[-1] == '/':
                print(x[:-1])
                safe.append([x[:-1]])
            else:
                print(x)
                safe.append([x])

        print(safe)
        with open('data/sources/trusted.csv', 'w+', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(safe)
