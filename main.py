import pandas as pd
from os import path
from nlpPipeline import batchProc
from claim import docClaim
from webcrawl import nlpFeed


def main():

    #Read in input claims
    df=pd.read_table(path.join("data","Politihop","Politihop_train.tsv"),sep='\t').head(30)

    #The input claims data has multiple repetitions of each text due to containing multiple verifiable claims. This
    #is handled later so for now the text must be de-duplicated. Other text pre-processing/cleansing occurs here.
    statementSet = set()
    for s in df['statement']:
        while not s[0].isalpha():
            s=s[1:]
        if s.split(" ")[0].lower() == "says":
            #todo instead, stick the person speaking on the front here
            s=(s.partition(" "))[1]
        #Allows for filtering to debug specific example.
        if (True or 'bastard' in s) and any(x !=" " for x in s):
            statementSet.add(s)

    docs = batchProc(statementSet)

    for doc in docs:
        input("press enter for next")
        try:
            tlClaim = docClaim(doc,isTLClaim=True)
        except NotImplementedError:
            continue
        """for subclaim in tlClaim.subclaims:
            queries = subclaim.kb.prepSearch()
            sources=[]
            for q in queries:
                sources.extend(nlpFeed(q))
            print("SOU", sources)
            for a in sources:
                if len(a) < 5 or 'cookies' in sources:
                    sources.remove(a)
            newData = batchProc(sources)

            for doc in newData:
                try:
                    tlEv = docClaim(doc)
                except NotImplementedError:
                    continue
                print("EVKB-S")
                for sc in tlEv.subclaims:
                    print(sc.kb)
                print("EVKB-E")
            #Make the new claims into the graph, then into logic.
"""


        print("...............")

    #Information extraction completed, with doc._.DCorefs storing coreferences, frames storing VSD, oieSubclaims
    #storing extracted relations. These are all shared among the top-level claim, which is now instantiated.
    #tlClaim = TLClaim(doc,oieSubclaims)
    #tlClaim.printTL()

if __name__ == '__main__':
    main()
