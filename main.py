import pandas as pd
from os import path

import evidence
from nlpPipeline import batchProc
from claim import docClaim
from webcrawl import nlpFeed, dumpToCache


def main(name=None):

    #Read in input claims
    df=pd.read_table(path.join("data","Politihop","Politihop_train.tsv"),sep='\t').head(200)

    #The input claims data has multiple repetitions of each text due to containing multiple verifiable claims. This
    #is handled later so for now the text must be de-duplicated. Other text pre-processing/cleansing occurs here.
    statementSet = set()
    for i, row in df.iterrows():
        s=row['statement']
        while not s[0].isalpha() or s[0] == " ":
            s=s[1:]
        if s.partition(" ")[0].lower() == "says":
            author = row['author'].replace("Speaker: ", "")
            if author in ['Facebook posts', 'Viral image']:
                s = s.partition(" ")[2]
            else:
                s= author +" s" + s[1:]
        #Allows for filtering to debug specific example.
        if (False or 'Kobe' in s) and any(x !=" " for x in s):
            statementSet.add(s)

    docs = batchProc(statementSet)

    for doc in docs:

        try:
            tlClaim = docClaim(doc)
        except NotImplementedError:
            continue
        for subclaim in tlClaim.subclaims:
            queries, entities, ncs = subclaim.kb.prepSearch()
            matchSet = set()
            matchSet.update(entities)
            matchSet.update(ncs)

            sources=[]
            for q in queries:
                sources.extend(nlpFeed(q))
            dumpToCache()
            evidence.receiveDoc(queries,matchSet,sources)
            input("next...")
            #newData = batchProc(sources)"""


            """for doc in newData:
                try:
                    tlEv = docClaim(doc)
                except NotImplementedError:
                    continue
                print("EVKB-S")
                for sc in tlEv.subclaims:
                    print(sc.kb)
                print("EVKB-E")
            #Make the new claims into the graph, then into logic."""



        print("...............")

if __name__ == '__main__':
    main()
