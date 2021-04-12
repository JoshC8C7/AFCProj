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
    df=pd.read_table(path.join("data","Politihop","Politihop_train.tsv"),sep='\t').head(200)

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
        if True or politiDict[t] == 1: # 'Virginia' in s: #politiDict[t] == 1:# and 'Russians' not in s:# and 'climate' in s: #or 'Cooper' in s or 'trillion' in s:
            statementSet.add(s)
        dateMap[s] = None
        truthDict[s] = politiDict[t]
    #print("TD:",truthDict)
    return statementSet, dateMap, truthDict


def liarInput():

    dateMap = {}
    #Read in input claims
    df=pd.read_table(path.join("data","liarliar","train.tsv"),sep='\t').head(200)

    #The input claims data has multiple repetitions of each text due to containing multiple verifiable claims. This
    #is handled later so for now the text must be de-duplicated. Other text pre-processing/cleansing occurs here.
    statementSet = set()
    for i, row in df.iterrows():
        s=row[2]
        t=row[1]
        while not s[0].isalpha() or s[0] == " ":
            s=s[1:]
        if s.partition(" ")[0].lower() == "says":
                s = s.partition(" ")[2]
        #Allows for filtering to debug specific example.
        #if True or any(x in s for x in ['ever','far this','finally','just','newly','now','one day','one time','repeatedly','then','when']) and any(x !=" " for x in s):
        if False or 'Wages are on the rise' in s : #or 'Cooper' in s or 'trillion' in s:
            statementSet.add(s)
        dateMap[s] = None
        truthDict[s] = politiDict[t]
    #print("TD:",truthDict)

    return statementSet, dateMap, truthDict

DATA_IMPORT = {'politihop':politihopInput, 'liarliar':liarInput}

def main(name='politihop',format=''):
    inputFunc = DATA_IMPORT[name]
    results = []

    statementSet, dateMap, truthDict = inputFunc()
    docs = batchProc(statementSet,dateMap)

    for doc in docs:
        print("CLAIM: ", doc)
        scLevelResults=[]
        try:
            tlClaim = docClaim(doc)
        except NotImplementedError:
            continue
        for subclaim in tlClaim.subclaims:
            print("SUBCLAIM: ", subclaim.roots)
            queries, ncs, entities = subclaim.kb.prepSearch()
            sources=[]
            if format=='pfOnly':
                sources.extend(nlpFeed(tlClaim.doc.text))
            else:
                for q in queries:
                    sources.extend(nlpFeed(q))
            evidence.processEvidence(subclaim,ncs,entities,sources)
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
            proportion = max(scLevelResults)
            print("Guessed Truth:", str(proportion), "  Ground Truth:", truthDict[doc.text])
            results.append((proportionToTruth(proportion), truthDict[doc.text]))
        else:
            results.append((5, truthDict[doc.text]))
    print(results)


def proportionToTruth(proportion):
    if -0.5 <= proportion <= 0.5:
        return 0
    if proportion > 0.5:
        return 1
    else:
        return -1

if __name__ == '__main__':
    main()
