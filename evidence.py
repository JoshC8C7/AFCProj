from string import punctuation
import spacy
from nltk.corpus import sentiwordnet as swn, wordnet as wn
from nltk.corpus.reader import WordNetError
from textacy import similarity
import claim


import json
import nlpPipeline
from spacy.lang.en import English, STOP_WORDS
import requests
endpoint_url = "https://www.wikidata.org/w/api.php?action=wbsearchentities&search={0}&language=en&format=json&limit=3"

nlp = English()
nlp.add_pipe(nlp.create_pipe("sentencizer"))
nlp1 = spacy.load('en_core_web_lg')
nlp1.disable_pipes()

with open('data/pb2wn.json','r') as inFile:
    pb2wn = json.load(inFile)

def wikidataCompare(e1, e2, cache):
    if all(x.pos_ not in ('PROPN','NOUN') for x in e1) or all(y.pos_ not in ('PROPN','NOUN') for y in e2) or e1.label_ != e2.label_:
        return False
    if e1.label_ is None:
        e1 = e1.root
        e2 = e2.root
    if e1.text in cache:
        k1 = cache[e1.text]
    else:
        print("polling: ", e1)
        k1 = requests.get(endpoint_url.format(e1.text.replace(" ", "%20"))).json().get('search', [])
        if k1:
            cache[e1.text] = k1
        else: return False

    if e2.text in cache:
        k2=cache[e2.text]
    else:
        print("Polling: ", e2)
        k2 = (x.get('id','') for x in requests.get(endpoint_url.format(e2.text.replace(" ", "%20"))).json().get('search',[]))
        if k2:
            cache[e2.text] = k2
        else: return False
    return any(x['id'] in k2 for x in k1)

def numCompare(e1,e2):
    #print(e1, e1.label_, e2, e2.label_)
    from word2number.w2n import word_to_num as w2n
    import re
    if e1.label_ == e2.label_ == 'CARDINAL':
        #print("E1:", e1)
        #print("E2:",e2)
        re1 = re.findall(r'[\d.]+',e1.text)
        re2 = re.findall(r'[\d.]+',e2.text)
        try:
            int1 = float(w2n(e1.text))*(1.0 if not re1 else float(re1[0]))
        except ValueError:
            if re1: int1= float(re1[0])
            else: return True
        try:
            int2 = float(w2n(e2.text))*(1.0 if not re2 else float(re2[0]))
        except ValueError:
            if re2: int2= float(re2[0])
            else: return True

        #print("checking ranges: ", int2*0.7, " < ", int1, " < ", int2*1.3, int2 * 0.7 < int1 < int2 * 1.3)
        return int2 * 0.7 < int1 < int2 * 1.3
    else:
        return True

def processEvidence(subclaim, ncs, entities, sources):
    wikiCache = {}
    strippedExisting = subclaim.doc.text.translate(str.maketrans('', '', punctuation)).lower()
    evidence = receiveDoc(sources, subclaim.doc)
    urlMap = dict((x[1],x[0]) for x in evidence)
    evDocs = nlpPipeline.batchProc(list(x[1] for x in evidence),{},urlMap,(ncs,entities))
    soughtV = set()
    kbElligibleOIEs = []

    for x in subclaim.kb.argBaseC.values():
        if x.uvi is not None:
            soughtV.add(x)
    for doc in evDocs:
        strippedIncoming = doc.text.translate(str.maketrans('', '', punctuation)).lower()
        if (strippedExisting == strippedIncoming):
            continue
        oieAccum = {}
        uviMatchedOies = {}
        for oie in doc._.OIEs:
            oieAccum[oie['V']] = oie
        #pprint(oieAccum)
        if oieAccum:
            proceeding, sentimentMatches = uviMatch(soughtV,oieAccum, doc._.Uvis)
            if proceeding:
                #print("UVIMATCHES: ", proceeding)
                for p in proceeding:
                    #Need to retrieve the OIE from the incoming, and the node or whatever for the new one.
                    #print("OIE from incoming:", p[1])
                    verbNode = subclaim.kb.getEnabledArgBaseC().get(p[0],None)
                    if verbNode is not None:
                        if verbNode in uviMatchedOies:
                            uviMatchedOies[verbNode].append(p[1])
                        else:
                            uviMatchedOies[verbNode] = [p[1]]
            if uviMatchedOies:
                nounMatch(uviMatchedOies,sentimentMatches,subclaim,doc,wikiCache)

    return kbElligibleOIEs

def corefCollect(span):
    corefs=[]
    for tok in span:
        corefs.extend((x.main for x in tok._.coref_clusters))
    return corefs

def nodeCompare(IargK,IargV,Espan,wikiCache):
    #print("COMPARING: ", IargV, " with ", Espan)
    #print("stage 1",end=' ')
    sim = compute_similarity(IargV, Espan.span)
    if IargK != 'V' and sim > 0.5 and len(IargV)*5 > len(Espan.span) > len(IargV)//5 and claim.getEdgeStyle(IargK, IargV) != 'dotted':
        Icorefs, ECorefs = corefCollect(IargV), corefCollect(Espan.span)


        if (not(IargV.ents or Espan.span.ents) and sim > 0.8):
            print("L1: ", IargV, " with ", Espan.span)
            return True
        else:
            c1, c2=0,0
            v = ((similarity.levenshtein(x.root.text, y.root.text) > 0.7 and len(y.root.text)*5 > len(x.root.text) > len(y.root.text)//5) for x in IargV.noun_chunks for y in
                 Espan.span.noun_chunks)
            for x in v:
                if x: c1+=1
                else: c2+=1
            if c1>c2:
                print("L2: ", IargV, " with ", Espan.span)
                return True
            elif IargV.ents and Espan.span.ents and similarity.levenshtein(IargV.text, Espan.span.text) > 0.3 and (any(numCompare(x,y) and (similarity.levenshtein(x.text, y.text) > 0.7 or wikidataCompare(x,y,wikiCache)) for x in IargV.ents+Icorefs for y in set(Espan.span.ents+ECorefs))):
                print("L3: ", IargV, " with ", Espan.span)
                return True
    #print(False)
        """if True or 'wrong direction' in IargV.text:
            print("Fail:", IargV, " with ", Espan.span)
            print("IARGVENTS?",IargV.ents, "/ ", "EARGVENTS", Espan.span.ents, "SIM ",sim)
            print("IARGNOUNCH",IargV.noun_chunks, "/", "Eargvnounch", Espan.span.noun_chunks)
            print("LEV ",similarity.levenshtein(IargV.text, Espan.span.text),(any(numCompare(x,y) and (similarity.levenshtein(x.text, y.text) > 0.7 or wikidataCompare(x,y,wikiCache)) for x in IargV.ents+Icorefs for y in set(Espan.span.ents+ECorefs))))
    """
    return False

def nouns2KB(oiesEnt, arity, verbT, args, kb, wikiCache, kbEnt, docIn, predNeg=False):
    #print("OIESENT ", oiesEnt)
    accum = [False] * int(arity)
    oieIter = list(sorted(oiesEnt.items()))
    for index, EargV in enumerate(args):
        Espan = kb.getEnabledArgBaseC().get(EargV, None)

        if Espan is None:
            # print("failed ", EargV)
            continue
        # print(verbT, "ARGS", EargV, Espan.span.ents, corefCollect(Espan.span))
        # print(list(xk+' '+xv.text+ ' '+ str(corefCollect(xv)) for xk,xv in oieIter))
        for IargK, IargV in oieIter:
            # print("VERB", verb, "IARGV:",IargV, "EARGV", EargV)
            if nodeCompare(IargK, IargV, Espan, wikiCache):
                accum[index] = True
                # print("spanSim:", fv.similarity(span.span), " incomingEnts:",fv.ents, " existingents:",span.span.ents)
                #print(index, ": ", IargV, "->", Espan.span)

    # print(accum, sum(1 for i in accum if i)/int(arity))
    if sum(1 for i in accum if i) / int(arity) >= 0.66:
        modifiers = []
        # print("SKB",subclaim.kb.kb2_args)
        if 'ARGM-NEG' in oiesEnt:
            predNeg = True
        for mod in kb.kb2_args.get(kbEnt.split(" ")[0],
                                            []):  # for this predicate, are there modifiers listed? iterate over them
            # note that some won't have assocaited modifier lists because they were internal nodes and so were added during establishrule,
            # rather than via conjestablish.
            enrichedModifierNode = kb.getEnabledArgBaseC().get(mod.split('(')[1].replace(')', ''),
                                                                        None)  # Are these modifiers enabled?
            if enrichedModifierNode is None:
                continue
            for IargK, IargV in oieIter:
                # print("IARGK: ", IargK)
                if nodeCompare(IargK, IargV, enrichedModifierNode, wikiCache):
                    modifiers.append(mod)
                    # print("spanSim:", fv.similarity(span.span), " incomingEnts:",fv.ents, " existingents:",span.span.ents)
                    # print("MOD FOUND:", IargV, "->", enrichedModifierNode.span)

        newArgs = []
        for index, ent in enumerate(accum):
            if True or ent:
                newArgs.append(args[index])
            else:
                fv = kb.getFreeVar()
                newArgs.append(fv)

        pred = verbT + arity + '(' + ','.join(newArgs) + ')'
        if predNeg:
            pred = '-' + pred
            #print("PREDNEGGED")

        kb.evidenceMap[pred] = docIn
        for mod in modifiers:
            pred += ' &' + mod
            kb.evidenceMap[mod] = docIn
        kb.addToKb(pred)
        #print("NEW EVIDENCE: ", pred, "----->", docIn, " @ ", docIn._.url)

    return

def nounMatch(oiesDict,sentimentMatches,subclaim,docIn,wikiCache):
    #print("OIEDICT",oiesDict)
    #print("EXISTING", subclaim.kb.kb2)
    for kbEnt in subclaim.kb.kb2: #Iterate over KB rules
        #print("KBENT",kbEnt)
        kbEnt = kbEnt.replace('-','')
        splitvarg = kbEnt.split('(')
        verbT,arity,args = splitvarg[0][:-1],splitvarg[0][-1:],splitvarg[1].split(')')[0].split(',')
        verb = subclaim.kb.getEnabledArgBaseC().get(verbT, None)
        for oiesEnt in oiesDict.get(verb,[]): #Iterate over incoming oies
            nouns2KB(oiesEnt,arity, verbT, args,subclaim.kb,wikiCache,kbEnt, docIn)

        for sentiOIE, sentiMatchDirection in sentimentMatches:
            nouns2KB(sentiOIE,arity,verbT,args,subclaim.kb,wikiCache,kbEnt,docIn,predNeg=sentiMatchDirection)
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


#Takes ALL existing nodes (before looking at all their UVis), and ALL incoming oies (ditto), and look for any uvi matches.
#Incoming oies are in a dict with key = verb span, value = ?
#Returns pairs of (ID of existing node that matches uvi, incoming OIE which matched it)
def uviMatch(existing, incoming, incomingUviMap):
    oieMatches = []
    sentimentMatches = []
    seenSent = set()
    for jk, j in incoming.items():
        jm = pb2wn.get(incomingUviMap[jk], None)
        if jm is not None:
            for jmm in jm:
                try:
                    wnj = wn.synset(jmm)
                except WordNetError:
                    continue
                for i in existing:
                    im = pb2wn.get(i.uvi,None)
                    if im is not None:
                        for imm in im:
                            try:
                                wni = wn.synset(imm)
                            except WordNetError:
                                continue
                            sim = (wn.path_similarity(wni,wnj))
                            #print(wni,wnj,sim)
                            if sim > 0.5 and (i.ID,j) not in oieMatches:
                                #print("OIEHIT", wni, "->", wnj, " : ", j)
                                #print(j['V'].doc._.coref_scores)
                                oieMatches.append((i.ID,j))

                if jk not in seenSent:
                    seenSent.add(jk)
                    wnj_s = swn.senti_synset(jmm)
                    if wnj == wn.synset("be.v.08") or (wnj_s.pos_score() >= 0.5 and wnj_s.neg_score() < 0.5):
                        sentimentMatches.append((j,True))
                    elif wnj_s.neg_score() >= 0.5 and wnj_s.pos_score() < 0.5:
                        sentimentMatches.append((j,False))

    #print("SENTIMATCHES ", sentimentMatches)

    return list(oieMatches), sentimentMatches


def receiveDoc(sources, docText):
    sentences = []
    for source in sources:
        url = source[0]
        s2 = ''.join(filter(lambda x: x in string.printable and x not in ['{','}'], source[1]))
        if len(s2) < 5:
            continue
        doc = nlp(s2)
        #filteredEnts = list(filter(lambda x: len(x) < 40, matchSet))
        for sent in doc.sents:
            #print(sent)
            #if any(len(find_near_matches(p,sent.text,max_l_dist=len(p)//3)) for p in filteredEnts):
            if len(sent) < 150 and sent.text.lower() not in docText.text.lower():
                sentences.append((url, sent.text.replace("\n","")))

    newSents=[]
    while len(sentences) % 2!= 0:
        sentences.append('')
    for i in range(0,len(sentences)-2,2):
        newSents.append((sentences[i][0],sentences[i][1] + ' ' + sentences[i+1][1]))

    return newSents

if __name__ == '__main__':
    from nltk.corpus import wordnet as wn
    while True:
        inp1 = wn.synset(input('enter word: '))
        for s in inp1:
            print(s)