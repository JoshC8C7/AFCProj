from pprint import pprint
from string import punctuation
import spacy
from fuzzysearch import find_near_matches
from nltk.corpus import wordnet as wn
import networkx as nx
from textacy import similarity
from textacy.similarity import levenshtein
import claim

import json
import nlpPipeline
nlp1 = spacy.load('en_core_web_lg')
from spacy.lang.en import English, STOP_WORDS

PARTIAL_TOLERANCE = 90
MATCH_SENSITIVITY = 1 #Decreasing this may increase accuracy, but will signifcantly increase run-time.

nlp = English()
nlp.add_pipe(nlp.create_pipe("sentencizer"))

with open('data/pb2wn.json','r') as inFile:
    pb2wn = json.load(inFile)

def processEvidence(subclaim, matchSet, sources):
    strippedExisting = subclaim.doc.text.translate(str.maketrans('', '', punctuation)).lower()
    evidence = receiveDoc(matchSet,sources)
    evDocs = nlpPipeline.batchProc(evidence,{},matchSet)
    soughtV = set()
    kbElligibleOIEs = []

    for x in subclaim.kb.argBaseC.values():
        if x.uvi is not None:
            soughtV.add(x)
    print("EXS", subclaim.kb.kb2)
    for doc in evDocs:
        strippedIncoming = doc.text.translate(str.maketrans('', '', punctuation)).lower()
        if (strippedExisting == strippedIncoming):
            continue
        oieAccum = {}
        uviMatchedOies = {}
        for oie in doc._.OIEs:
            oieAccum[oie['V']] = oie
        if oieAccum:
            #print("DOC",doc,"OIEACCUM",oieAccum)
            proceeding, sentimentMatches = uviMatch(soughtV,oieAccum, doc._.Uvis)
            if proceeding:
                #print(subclaim.doc, "-> ", doc)
                for p in proceeding:
                    #Need to retrieve the OIE from the incoming, and the node or whatever for the new one.
                    #print("OIE from incoming:", p[1])
                    verbNode = subclaim.kb.getEnabledArgBaseC().get(p[0],None)
                    if verbNode is not None:
                        uviMatchedOies[verbNode] = p[1]
            if uviMatchedOies:
                nounMatch2(uviMatchedOies,sentimentMatches,subclaim)

    return kbElligibleOIEs

#Takes oiesdict (key = the argNode corresponding to the matched verb, value = the oie it matched with) and subclaim.
def nounMatch(oiesDict, subclaim):
    nounMatch2(oiesDict,subclaim)
    substitutedOIES = []
    from textacy import similarity
    for ek, ev in oiesDict.items(): #For each OIE
        seen=[]
        modFlag = False
        #print("EK",ek,"EV",ev)
        substitutedDict = ev.copy()
        for neighb in subclaim.graph.in_edges(nbunch=ek.ID):
            span = subclaim.kb.getEnabledArgBaseC().get(neighb[0], None)
            if span is not None:
                for fk, fv in ev.items(): #for all args of the verb
                    #print(fv, fv.ents, "??", span.span, span.span.ents, compute_similarity(fv,span.span), claim.getEdgeStyle(fk,fv))
                    if fk != 'V' and span.span not in seen and compute_similarity(fv,span.span) > 0.5 and claim.getEdgeStyle(fk,fv) != 'dotted':
                        if ((not fv.ents and not span.span.ents) and compute_similarity(fv,span.span) > 0.7) or (fv.ents and span.span.ents and (any(similarity.levenshtein(x.text, y.text) > 0.5 for x in fv.ents for y in span.span.ents))):
                            modFlag = True
                            substitutedDict[fk] = span
                            seen.append(span.span)
                            #print("spanSim:", fv.similarity(span.span), " incomingEnts:",fv.ents, " existingents:",span.span.ents)
                            print(fv, "->",span.span)


        if modFlag:
            substitutedDict['V'] = ek
            substitutedOIES.append(substitutedDict)
    return substitutedOIES


def corefCollect(span):
    corefs=[]
    for tok in span:
        corefs.extend((x.main for x in tok._.coref_clusters))
    return corefs

def nounMatch2(oiesDict,subclaim):
    print("OIESIN",oiesDict)

    for kbEnt in subclaim.kb.kb2:
        #print("KBENT",kbEnt)
        seen = []
        predNeg = False
        kbEnt = kbEnt.replace('-','')
        splitvarg = kbEnt.split('(')
        verbT,arity,args = splitvarg[0][:-1],splitvarg[0][-1:],splitvarg[1].split(')')[0].split(',')
        verb = subclaim.kb.getEnabledArgBaseC().get(verbT, None)
        accum = [False] * int(arity)
        if verb in oiesDict:
            for index, EargV in enumerate(args):
                Espan = subclaim.kb.getEnabledArgBaseC().get(EargV, None)

                if Espan is None:
                    print("failed ", EargV)
                    continue
                print(verbT, "ARGS", EargV, Espan.span.ents, corefCollect(Espan.span))
                for IargK, IargV in sorted(oiesDict[verb].items()):
                    #print("VERB", verb, "IARGV:",IargV, "EARGV", EargV)
                    if IargK in ['ARG0','ARG1','ARG2','ARG3','ARG4','ARG5']+['DIR', 'PRD','MNR']:
                        if IargK != 'V' and Espan.span not in seen and compute_similarity(IargV,Espan.span) > 0.5 and claim.getEdgeStyle(IargK, IargV) != 'dotted':
                            Icorefs, ECorefs = corefCollect(IargV), corefCollect(Espan.span)
                            if ((not IargV.ents and not Espan.span.ents) and compute_similarity(IargV, Espan.span) > 0.7) \
                                    or (IargV.ents and Espan.span.ents and (any(similarity.levenshtein(x.text, y.text) > 0.5 for x in IargV.ents for y in Espan.span.ents))) \
                                    or (Icorefs and ECorefs and (any(similarity.levenshtein(x.text, y.text) > 0.5 for x in IargV.ents for y in Espan.span.ents))):
                                seen.append(Espan.span)
                                accum[index] = True
                                # print("spanSim:", fv.similarity(span.span), " incomingEnts:",fv.ents, " existingents:",span.span.ents)
                                print(index,": ",IargV, "->", Espan.span)



        if all(accum):
            modifiers=[]
            for mod in subclaim.kb.kb2_args.get(kbEnt,[]): #for this predicate, are there modifiers listed? iterate over them
                #note that some won't have assocaited modifier lists because they were internal nodes and so were added during establishrule,
                #rather than via conjestablish.
                enrichedModifierNode = subclaim.kb.getEnabledArgBaseC()[mod] #Are these modifiers enabled?
                if enrichedModifierNode is None:
                    continue
                modType = 'ARGM-'+mod[:3] #Their type is modType
                if modType in oiesDict[kbEnt]: #See if this modType is in the incoming oies
                    if similarity.levenshtein(oiesDict[kbEnt][modType].text, enrichedModifierNode.span.text) > 0.5:
                        modifiers.append(mod)


            newArgs = []
            for index, ent in enumerate(accum):
                if ent:
                    newArgs.append(args[index])
                else:
                    fv = subclaim.kb.getFreeVar()
                    newArgs.append(fv)

            pred = verbT+arity+'('+','.join(newArgs) + ')'
            if predNeg:
                pred = '-'+pred

            for mod in modifiers:
                pred += ' &' + mod
            subclaim.kb.addToKb(pred)

    return

def checkGrammatical(inSpan):
    return all((tok.pos_ in claim.grammaticalPOS or tok.tag_ in claim.grammaticalTAG) for tok in inSpan)

def compute_similarity(doc1, doc2):
    import numpy as np
    vector1 = np.zeros(300)
    vector2 = np.zeros(300)
    for token in doc1:
        if (token.text not in STOP_WORDS):
            vector1 = vector1 + token.vector
    vector1 = np.divide(vector1, len(doc1))
    for token in doc2:
        if (token.text not in STOP_WORDS):
            vector2 = vector2 + token.vector
    vector2 = np.divide(vector2, len(doc2))
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

from nltk.corpus import sentiwordnet as swn

#Takes ALL existing nodes (before looking at all their UVis), and ALL incoming oies (ditto), and look for any uvi matches.
#Incoming oies are in a dict with key = verb span, value = ?
#Returns pairs of (ID of existing node that matches uvi, incoming OIE which matched it)
def uviMatch(existing, incoming, incomingUviMap):
    oieMatches = []
    sentimentMatches = []
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
                        if (imm,jmm) not in seen:
                            seen.add((imm,jmm))
                            try:
                                wnj = wn.synset(jmm)
                            except:
                                continue
                            wnj_s = swn.senti_synset(jmm)
                            if wnj_s.pos_score() >= 0.5:
                                sentimentMatches.append((j, True))
                            elif wnj_s.neg_score() >= 0.5:
                                sentimentMatches.append((j,False))

                            sim = (wn.path_similarity(wni,wnj))
                            #print(wni,wnj,sim)
                            if sim > 0.5 and (i.ID,j) not in oieMatches:
                                oieMatches.append((i.ID,j))
    return list(oieMatches), sentimentMatches


def receiveDoc(matchSet, sources):
    sentences,matched = [], {}
    for s in sources:
        if len(s) < 5:
            continue
        doc = nlp(s)
        matchThreshold = 1#max(int(MATCH_SENSITIVITY * len(matchSet))//3,1)
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
    """newSents=[]
    if len(sentences) % 4 != 0:
        sentences.append('')
    for i in range(0,len(sentences)-4,4):
        newSents.append(sentences[i] + ' ' + sentences[i+1] +' ' + sentences[i+2] + ' ' + sentences[i+3])"""
    return sentences

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
