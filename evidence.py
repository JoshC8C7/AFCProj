import spacy
from fuzzysearch import find_near_matches


nlp1 = spacy.load('en_core_web_lg')
from spacy.lang.en import English
PARTIAL_TOLERANCE = 90
MATCH_SENSITIVITY = 1 #Decreasing this may increase accuracy, but will signifcantly increase run-time.

nlp = English()
nlp.add_pipe(nlp.create_pipe("sentencizer"))


def processEvidence(subclaim, matchSet, sources):
    from nlpPipeline import batchProc
    evidence = receiveDoc(matchSet,sources)
    print("EV", evidence)
    evDocs = batchProc(evidence)
    oieAccum=[]
    for doc in evDocs:
        for oie in doc._.OIEs:
            for arg in oie.values():
                if any(ent.text in matchSet for ent in arg.ents):
                    oieAccum.append(oie)
                    break

    for x in oieAccum:
        print(x)

    #Now send the evidence docs through the separate logic pipeline, feeding into the KB of subclaim,
    #then return to main.py to evaluate what extra evidence is needed.

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
