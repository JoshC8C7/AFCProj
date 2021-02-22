import spacy
from fuzzysearch import find_near_matches
from nltk.corpus import wordnet as wn
import networkx as nx
from textacy.similarity import levenshtein

import json
import nlpPipeline
nlp1 = spacy.load('en_core_web_lg')
from spacy.lang.en import English
PARTIAL_TOLERANCE = 90
MATCH_SENSITIVITY = 1 #Decreasing this may increase accuracy, but will signifcantly increase run-time.

nlp = English()
nlp.add_pipe(nlp.create_pipe("sentencizer"))

with open('data/pb2wn.json','r') as inFile:
    pb2wn = json.load(inFile)

def processEvidence(subclaim, matchSet, sources):
    evidence = receiveDoc(matchSet,sources)
    evDocs = nlpPipeline.batchProc(evidence,{},matchSet)
    soughtV = set()

    for x in subclaim.kb.argBaseC.values():
        if x.uvi is not None:
            soughtV.add(x)

    for doc in evDocs:
        oieAccum = {}
        uviMatchedOies = {}
        for oie in doc._.OIEs:
            oieAccum[oie['V']] = oie
        if oieAccum:
            proceeding = uviMatch(soughtV,oieAccum, doc._.Uvis)
            if proceeding:
                #print(subclaim.doc, "-> ", doc)
                for p in proceeding:
                    #Need to retrieve the OIE from the incoming, and the node or whatever for the new one.
                    print("OIE from incoming:", p[1])
                    verbNode = subclaim.kb.getEnabledArgBaseC()[p[0]]
                    if verbNode is not None:
                        uviMatchedOies[verbNode] = p[1]
            nounMatch(uviMatchedOies,subclaim)

                #Run node inference
                #form the logic to dump it in.

        else:
            print("Miss")

#Takes oiesdict (key = the argNode corresponding to the matched verb, value = the oie it matched with) and subclaim.
def nounMatch(oiesDict, subclaim):
    substitutedOIES = []
    for ek, ev in oiesDict.items():
        substitutedDict = ev.copy()
        for neighb in subclaim.graph.in_edges(nbunch=ek.ID):
            span = subclaim.kb.getEnabledArgBaseC().get(neighb[0], None)
            if span is not None:
                for fk, fv in ev.items():
                    if fv.similarity(span.span) > 0.5 and any(x.similarity(y) > 0.5 for x in fv.ents for y in span.span.ents):
                        substitutedDict[fk] = span.span
                        print(fk, "->",span.span)

        if substitutedDict != ev:
            substitutedOIES.append(substitutedDict)

# todo args connected to this argbaseC.


#Takes ALL existing nodes (before looking at all their UVis), and ALL incoming oies (ditto), and look for any uvi matches.
#Incoming oies are in a dict with key = verb span, value = ?
#Returns pairs of (ID of existing node that matches uvi, incoming OIE which matched it)
def uviMatch(existing, incoming, incomingUviMap):
    oieMatches = []
    seen = set()
    for i in existing:
        for jk, j in incoming.items():
            im = pb2wn.get(i.uvi,None)
            jm = pb2wn.get(incomingUviMap[jk],None)
            if im is not None and jm is not None:
                for imm in im:
                    try:
                        wni = wn.synset(imm)
                    except:
                        continue
                    for jmm in jm:
                        if jmm not in seen:
                            seen.add(jmm)
                            try:
                                wnj = wn.synset(jmm)
                            except:
                                continue
                            sim = (wn.path_similarity(wni,wnj))
                            #print(wni,wnj,sim)
                            if sim > 0.5:
                                oieMatches.append((i.ID,j))

    return oieMatches


def receiveDoc(matchSet, sources):
    sentences,matched = [], {}
    for s in sources:
        if len(s) < 5:
            continue
        doc = nlp(s)
        matchThreshold = max(int(MATCH_SENSITIVITY * len(matchSet))//3,1)
        for ent in matchSet:
            for sent in doc.sents:
                mat = find_near_matches(ent, sent.text, max_l_dist=len(ent)//3)
                if mat:
                    if sent.text.encode("ascii", errors="ignore").decode() == sent.text:

                        #Need to match with MATCH_SENSITIVITY+ entities, so for default (MATCH_SENSITIVITY = 2):
                        #After matching with 1, move it to the 'matched' list, then ig it matches again,
                        #move it into the final (sentences) list and set its entry in matched to > MATCH_SENSITIVITY (3)
                        #such that it cannot continue to be added to sentences (duplicates).
                        if sent.text in matched:
                            if matched[sent.text] == matchThreshold:
                                sentences.append(sent.text.replace("\n",""))
                                matched[sent.text] += 1
                            #else None
                        else:
                            matched[sent.text] = 1
                            if matchThreshold == 1:
                                sentences.append(sent.text.replace("\n", ""))
                                matched[sent.text] += 1
                            #else None

        """#Method 2: fuzzywuzzy on the ncs + ents

        doc1 = nlp1(s)
        vals1 = list(matchSet)
        #vals1.extend(allMatches)
        for sent in doc1.sents:
            vals2 = list(x.text for x in sent.ents)
            #vals2.extend(y.text for y in sent.noun_chunks)
            for val1 in vals1:
                for val2 in vals2:
                    if partial_ratio(val1,val2) > PARTIAL_TOLERANCE:
                        if sent.text.encode("ascii", errors="ignore").decode() == sent.text and sent.text.replace("\n", "") not in sentences2:
                            sentences2.append(sent.text.replace("\n", ""))"""
    return sentences

def find_sim(tok1, tok2):
    w1 = wn.synsets()


if __name__ == '__main__':
    from nltk.corpus import wordnet as wn
    while True:
        inp1 = wn.synsets(input('enter word: '))
        for s in inp1:
            print(s)



    #ADJ, ADJ_SAT, ADV, NOUN, VERB = "a", "s", "r", "n", "v"
    """
    inp1 = wn.synsets(input('Enter word1: '),pos=wn.NOUN)
    inp2 = wn.synsets(input('Enter word2: '),pos=wn.NOUN)
    for s in inp1:
        for s2 in inp2:
            print(s.definition(), "   /    ", s2.definition(), "   /   ", wn.path_similarity(s, s2))"""
