import pandas as pd
from os import path
import evidence
from nlpPipeline import batchProc
from claim import docClaim
from webcrawl import nlpFeed

#Politihop-to-standard label conversion, as in Politihop.
politiDict = {'true':1,'mostly-true':1,'barely-true':-1,'half-true':0,'mostly-false':-1,'pants-fire':-1,'false':-1}
truthDict = {}

#Handles input for politihop
def politihopInput(data):

    #Read in input claims
    df=pd.read_table(path.join("data","Politihop",data),sep='\t').head(200)

    #The input claims data has multiple repetitions of each text due to containing multiple verifiable claims. This
    #is handled later so for now the text must be de-duplicated. Other text pre-processing/cleansing occurs here.
    statementSet = set()
    for i, row in df.iterrows():
        s=row['statement']
        t=row['politifact_label']
        while not s[0].isalpha() or s[0] == " ":
            s=s[1:]

        #Push in name of author to claim where a real person i.e. not a viral image/post
        if s.partition(" ")[0].lower() == "says":
            author = row['author'].replace("Speaker: ", "")
            if True or author in ['Facebook posts', 'Viral image']:
                s = s.partition(" ")[2]
            else:
                s= author +" s" + s[1:]

        if True or politiDict[t] == 1:
            statementSet.add(s)
        truthDict[s] = politiDict[t]
    #print("TD:",truthDict)
    return statementSet, truthDict


def liarInput(data):
    #Read in input claims
    df=pd.read_table(path.join("data","liarliar",data),sep='\t').head(200)

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
        statementSet.add(s)
        truthDict[s] = politiDict[t]

    return statementSet, truthDict

#Change which dataset to import
DATA_IMPORT = {'politihop':politihopInput, 'liarliar':liarInput}

def processClaim(doc, limiter):
    print("CLAIM: ", doc)
    scLevelResults = []

    #Split claim into subclaim and then logical formulae
    tlClaim = docClaim(doc)

    #Iterate through generated subclaims and attempt to prove.
    for subclaim in tlClaim.subclaims:

        #Obtain search terms from processed subclaims
        queries, ncs, entities = subclaim.kb.prepSearch()

        #Collect evidence from webcrawler, modified by limiter as appropriate.
        sources = []
        if limiter is not None and limiter == 'pfOnly':
            sources.extend(nlpFeed(tlClaim.doc.text))
        else:
            for q in queries:
                sources.extend(nlpFeed(q))

        #Process collected evidence into knowledge base.
        evidence.processEvidence(subclaim, ncs, entities, sources)

        #Attempt proof
        result = subclaim.kb.prove()
        print("RESULTAT", subclaim.doc, result)

        #Write result back, 1 if true, -1 if false/no proof found*, or error symbol '5'.
        if result is not None:
            scLevelResults.append(1 if result else -1)
        else:
            scLevelResults.append(5)

    #Determine overall document result from subclaim results.
    if scLevelResults:
        print("sc", scLevelResults)
        proportion = max(scLevelResults)
        print("Guessed Truth:", str(proportion), "  Ground Truth:", truthDict[doc.text])
        return ((proportion, truthDict[doc.text]))
    else:
        return ((5, truthDict[doc.text]))




def main(name='politihop',data='Politihop_train.tsv',limiter=None):
    results = []
    #Read in statements & associated Ground truth
    statementSet, truthDict = DATA_IMPORT[name](data)

    #Batch-process spaCy on documents
    docs = batchProc(statementSet)

    #Convert doc to claim, run inference, append results to list
    for doc in docs:
        results.append(processClaim(doc, limiter))

    #Return final results
    print(results)


if __name__ == '__main__':
    main()
