from string import punctuation, printable
from nltk.corpus import wordnet as wn
from nltk.corpus.reader import WordNetError
from textacy import similarity
from word2number.w2n import word_to_num as w2n
import re
import claim
import numpy as np

import json
import nlpPipeline
from spacy.lang.en import English, STOP_WORDS
import requests


#Setup wikidata API
endpoint_url = "https://www.wikidata.org/w/api.php?action=wbsearchentities&search={0}&language=en&format=json&limit=3"

# Import spacy sentencizer only for pre-processing/lightweight culling of irrelevant documents.
nlp = English()
nlp.add_pipe(nlp.create_pipe("sentencizer"))

#Read in PropBank to WordNet dictionary
with open('data/pb2wn.json', 'r') as inFile:
    pb2wn = json.load(inFile)

#Compare two entities via wikidata query. Checks for any intersection between two sets of 3 entities (UIDs) returned
#by a wikidata query, and caches them due to the high chance of a query being repeated.
def wikidataCompare(e1, e2, cache):
    #Only compare entities where at least one token is labelled as a (proper) noun.
    if all(x.pos_ not in ('PROPN', 'NOUN') for x in e1) or all(
            y.pos_ not in ('PROPN', 'NOUN') for y in e2) or e1.label_ != e2.label_:
        return False

    #If searching on noun chunks instead, just look at their root (which will be the noun).
    if e1.label_ is None:
        e1 = e1.root
        e2 = e2.root

    #Implement caching, make request to wikidata.
    if e1.text in cache and cache[e1.text] is not None:
        k1 = cache[e1.text]
    else:
        k1 = requests.get(endpoint_url.format(e1.text.replace(" ", "%20"))).json().get('search', [])
        if k1 is None:
            cache[e1.text] = None
            return False
        else:
            cache[e1.text] = k1

    #Repeat for 2nd entity
    if e2.text in cache and cache[e2.text] is not None:
        k2 = cache[e2.text]
    else:
        k2 = requests.get(endpoint_url.format(e2.text.replace(" ", "%20"))).json().get('search', [])
        if k2 is None:
            cache[e2.text] = None
            return False
        else:
            cache[e2.text] = k2

    #Return if any common entity
    return any(x['id'] in k2 for x in k1)

#Compare two entities which are numbers
def numCompare(e1, e2):

    if e1.label_ == e2.label_ == 'CARDINAL':
        re1 = re.findall(r'[\d.]+', e1.text)
        re2 = re.findall(r'[\d.]+', e2.text)

        #Attempt text to digit conversion (one million -> 1,000,000)
        try:
            int1 = float(w2n(e1.text)) * (1.0 if not re1 else float(re1[0]))
        except ValueError:
            if re1:
                int1 = float(re1[0])
            else:
                return True
        try:
            int2 = float(w2n(e2.text)) * (1.0 if not re2 else float(re2[0]))
        except ValueError:
            if re2:
                int2 = float(re2[0])
            else:
                return True

        #Accept comparison if within 30% either way.
        return int2 * 0.7 < int1 < int2 * 1.3
    else:
        return True

#Process incoming evidence into the knowledge base, if relevant. Attempts to co-resolve with entities
#already in the knowledge base, where possible.
def processEvidence(subclaim, ncs, entities, sources):
    wikiCache = {}

    #Strip existing to compare with incoming - i.e. don't accept a direct copy of the claim being checked, as evidence.
    strippedExisting = subclaim.doc.text.translate(str.maketrans('', '', punctuation)).lower()

    #Pre-process evidence
    evidence = receiveDoc(sources, subclaim.doc)
    urlMap = dict((x[1], x[0]) for x in evidence)

    #Batch run evidence through NLP pipeline
    evDocs = nlpPipeline.batch_proc(list(x[1] for x in evidence), urlMap, (ncs, entities))

    soughtV = set()
    kbElligibleOIEs = []

    #Obtain list of verbs to co-resolve with
    for x in subclaim.kb.argBaseC.values():
        if x.uvi is not None:
            soughtV.add(x)

    for doc in evDocs:
        strippedIncoming = doc.text.translate(str.maketrans('', '', punctuation)).lower()
        if (strippedExisting == strippedIncoming):
            continue

        #Extract verb frames to be passed to verb sense comparison function
        oieAccum = {}
        uviMatchedOies = {}
        for oie in doc._.OIEs:
            oieAccum[oie['V']] = oie

        #Co-resolve two possible verbs, if possible.
        if oieAccum:
            proceeding = uviMatch(soughtV, oieAccum, doc._.Uvis)
            if proceeding:
                for p in proceeding:
                    #Obtain rich arg node rather than just the ID, then form dictionary with key as the verb node, and
                    #values as a list of incoming relations whose verb has co-resolved with the key verb node.
                    verbNode = subclaim.kb.getEnabledArgBaseC().get(p[0], None)
                    if verbNode is not None:
                        if verbNode in uviMatchedOies:
                            uviMatchedOies[verbNode].append(p[1])
                        else:
                            uviMatchedOies[verbNode] = [p[1]]
            if uviMatchedOies:
                #If any verb matches are found, attempt to resolve nouns.
                nounMatch(uviMatchedOies, subclaim, doc, wikiCache)

    return kbElligibleOIEs

#Obtain coreferences for the input span
def corefCollect(span):
    corefs = []
    for tok in span:
        corefs.extend((x.main for x in tok._.coref_clusters))
    return corefs

#Compare two nouns on a mix of metrics.
def nodeCompare(IargK, IargV, Espan, wikiCache):
    sim = compute_similarity(IargV, Espan.span)
    #Baseline embedding similarity required, and cannot include nodes with an argument type that's dotted (excluded)
    if IargK != 'V' and sim > 0.5 and claim.get_edge_style(IargK, IargV) != 'dotted':
        Icorefs, ECorefs = corefCollect(IargV), corefCollect(Espan.span)

        #If No entities in either noun, then compare if embeddings are close (by cosine similarity).
        if (not (IargV.ents or Espan.span.ents) and sim > 0.8):
            return True
        else:
            #Check if more noun-chunks are alike than are dissimilar, based on levenshtein.
            c1, c2 = 0, 0
            v = ((similarity.levenshtein(x.root.text, y.root.text) > 0.7 and len(y.root.text) * 5 > len(
                x.root.text) > len(y.root.text) // 5) for x in IargV.noun_chunks for y in
                 Espan.span.noun_chunks)
            for x in v:
                if x:
                    c1 += 1
                else:
                    c2 += 1
            if c1 > c2:
                return True
            #If entities are present in both, require sane length and for numbers to be sufficiently similar (or the
            # comparison return True as the entities passed are not elligible), and one of levenshtein or wikidata
            # coresolution to exhibit sufficient similarity.
            elif IargV.ents and Espan.span.ents and len(IargV.ents) * 5 > len(
                Espan.span.ents) > len(IargV.ents) // 5 and (any(numCompare(x, y) and (
                    similarity.levenshtein(x.text, y.text) > 0.7 or wikidataCompare(x, y, wikiCache)) for x in
                                                         IargV.ents + Icorefs for y in set(Espan.span.ents + ECorefs))):
                return True

    #If not matched on any criteria, then return False.
    return False


def nounMatch(oiesDict, subclaim, docIn, wikiCache):
    for kbEnt in subclaim.kb.kb_rules_only:  # Iterate over KB rules
        kbEnt = kbEnt.replace('-', '')
        splitvarg = kbEnt.split('(')
        verbT, arity, args = splitvarg[0][:-1], splitvarg[0][-1:], splitvarg[1].split(')')[0].split(',')
        verb = subclaim.kb.getEnabledArgBaseC().get(verbT, None)
        for oiesEnt in oiesDict.get(verb, []):  # Iterate over incoming oies
            predNeg = False
            accum = [False] * int(arity)
            oieIter = list(sorted(oiesEnt.items()))
            for index, EargV in enumerate(args):
                Espan = subclaim.kb.getEnabledArgBaseC().get(EargV, None)

                if Espan is None:
                    continue
                for IargK, IargV in oieIter:
                    if nodeCompare(IargK, IargV, Espan, wikiCache):
                        accum[index] = True

            if sum(1 for i in accum if i) / int(arity) >= 0.66:
                modifiers = []
                if 'ARGM-NEG' in oiesEnt:
                    predNeg = True
                for mod in subclaim.kb.kb_rules_only_to_args.get(kbEnt.split(" ")[0],
                                                        []):  # for this predicate, are there modifiers listed? iterate over them
                    # note that some won't have assocaited modifier lists because they were internal nodes and so were added during establishrule,
                    # rather than via conjestablish.
                    enrichedModifierNode = subclaim.kb.getEnabledArgBaseC().get(mod.split('(')[1].replace(')', ''),
                                                                       None)  # Are these modifiers enabled?
                    if enrichedModifierNode is None:
                        continue
                    for IargK, IargV in oieIter:

                        if nodeCompare(IargK, IargV, enrichedModifierNode, wikiCache):
                            modifiers.append(mod)

                newArgs = []
                for index, ent in enumerate(accum):
                    if True or ent:
                        newArgs.append(args[index])
                    else:
                        fv = subclaim.kb.getFreeVar()
                        newArgs.append(fv)

                pred = verbT + arity + '(' + ','.join(newArgs) + ')'
                if predNeg:
                    pred = '-' + pred

                subclaim.kb.evidenceMap[pred] = docIn
                for mod in modifiers:
                    pred += ' &' + mod
                    subclaim.kb.evidenceMap[mod] = docIn
                subclaim.kb.addToKb(pred)
                print("NEW EVIDENCE: ", pred, "----->", docIn, " @ ", docIn._.url)

    return

#todo give credit for this !
def compute_similarity(doc1, doc2):
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


# Takes ALL existing nodes (before looking at all their UVis), and ALL incoming oies (ditto), and look for any uvi matches.
# Returns pairs of (ID of existing node that matches uvi, incoming OIE which matched it)
def uviMatch(existing, incoming, incomingUviMap):
    oieMatches = []
    for jk, j in incoming.items():
        jm = pb2wn.get(incomingUviMap[jk], None)
        if jm is not None:
            for jmm in jm:
                try:
                    wnj = wn.synset(jmm)
                except WordNetError:
                    continue
                for i in existing:
                    im = pb2wn.get(i.uvi, None)
                    if im is not None:
                        for imm in im:
                            try:
                                wni = wn.synset(imm)
                            except WordNetError:
                                continue
                            sim = (wn.path_similarity(wni, wnj))
                            if sim > 0.5 and (i.ID, j) not in oieMatches:
                                oieMatches.append((i.ID, j))


    return list(oieMatches)

#Preprocess input docs
def receiveDoc(sources, docText):
    sentences = []
    for source in sources:
        url = source[0]
        #Filter out non-printable characters.
        s2 = ''.join(filter(lambda x: x in printable and x not in ['{', '}'], source[1]))
        if len(s2) < 5:
            continue
        #Sentencize to split doc up and allow for granular filtering, remove extra long sentences.
        doc = nlp(s2)
        for sent in doc.sents:
            if len(sent) < 150 and sent.text.lower() not in docText.text.lower():
                sentences.append((url, sent.text.replace("\n", "")))

    #Rejoin sentences into pairs as to promote anaphora resolving more often.
    newSents = []
    while len(sentences) % 2 != 0:
        sentences.append('')
    for i in range(0, len(sentences) - 2, 2):
        newSents.append((sentences[i][0], sentences[i][1] + ' ' + sentences[i + 1][1]))

    return newSents
