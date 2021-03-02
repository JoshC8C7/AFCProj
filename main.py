from pprint import pprint

import pandas as pd
from os import path

import evidence
from nlpPipeline import batchProc
from claim import docClaim
from webcrawl import nlpFeed, dumpToCache
politiDict = {'true':1,'mostly-true':1,'barely-true':-1,'half-true':0,'mostly-false':-1,'pants-fire':-1,'false':-1}
truthDict = {}

def politihopInput():

    dateMap = {}
    #Read in input claims
    df=pd.read_table(path.join("data","Politihop","Politihop_train.tsv"),sep='\t').head(30)

    #The input claims data has multiple repetitions of each text due to containing multiple verifiable claims. This
    #is handled later so for now the text must be de-duplicated. Other text pre-processing/cleansing occurs here.
    statementSet = set()
    for i, row in df.iterrows():
        s=row['statement']
        t=row['politifact_label']
        while not s[0].isalpha() or s[0] == " ":
            s=s[1:]
        if s.partition(" ")[0].lower() == "says":
            author = row['author'].replace("Speaker: ", "")
            if True or author in ['Facebook posts', 'Viral image']:
                s = s.partition(" ")[2]
            else:
                s= author +" s" + s[1:]
        #Allows for filtering to debug specific example.
        #if True or any(x in s for x in ['ever','far this','finally','just','newly','now','one day','one time','repeatedly','then','when']) and any(x !=" " for x in s):
        if False or 'Cooper' in s or 'Mekong' in s or 'trillion' in s:
            statementSet.add(s)
        dateMap[s] = None
        truthDict[s] = politiDict[t]
    print("TD:",truthDict)
    return statementSet, dateMap, truthDict


DATA_IMPORT = {'politihop':politihopInput}

def main(name='politihop'):
    correct = 0
    incorrect =0
    inputFunc = DATA_IMPORT[name]

    statementSet, dateMap, truthDict = inputFunc()
    docs = batchProc(statementSet,dateMap)

    for doc in docs:
        scLevelResults=[]

        try:
            tlClaim = docClaim(doc)
        except NotImplementedError:
            continue
        for subclaim in tlClaim.subclaims:
            queries, ncs = subclaim.kb.prepSearch()
            sources=[]
            for q in queries:
                sources.extend(nlpFeed(q))
            dumpToCache()
            evidence.processEvidence(subclaim,ncs,sources)
            #subclaim.kb.OIEStoKB(logicReadyOIES)
            result = subclaim.kb.prove()
            print("RESULTAT", subclaim.doc, result)
            if result is not None:
                scLevelResults.append(1 if result else -1)
            else:
                scLevelResults.append(0)
            #input("next...")
        if scLevelResults:
            print("sc",scLevelResults)
            proportion = sum(scLevelResults)/len(scLevelResults)
            print("Guessed Truth:", str(proportion), "  Ground Truth:", truthDict[doc.text])
            if (proportionToTruth(proportion) == truthDict[doc.text]):
                correct+=1
            else:
                incorrect+=1
        else:
            incorrect+=1

    print("correct:",correct, "incorrect:",incorrect)

def proportionToTruth(proportion):
    if -0.5 <= proportion <= 0.5:
        return 0
    if proportion > 0.5:
        return 1
    else:
        return -1

if __name__ == '__main__':
    main()
