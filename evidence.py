import spacy
from fuzzysearch import find_near_matches
from fuzzywuzzy.fuzz import partial_ratio
from pprint import pprint

nlp1 = spacy.load('en_core_web_lg')
from spacy.lang.en import English
PARTIAL_TOLERANCE = 90

nlp = English()
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)

def receiveDoc(queries, matchSet, sources):
    sentences,sentences2 = [], []
    print("/////////////////////")
    matched = {}
    for s in sources:
        if len(s) < 5:
            continue
        doc = nlp(s)
        #Method 1: fuzzysearch
        for ent in matchSet:
            print(matched)
            for sent in doc.sents:
                mat = find_near_matches(ent, sent.text, max_l_dist=len(ent)//3)
                if mat:
                    if sent.text.encode("ascii", errors="ignore").decode() == sent.text:

                        #Need to match with 2+ entities. So when matching with 1, move it to the 'matched' list, then
                        #when it matches again, move it into the final (sentences) list and set its entry in matched to
                        #false, such that it cannot continue to be added to sentences (duplicates).
                        if sent.text in matched:
                            if matched[sent.text]:
                                sentences.append(sent.text.replace("\n",""))
                                matched[sent.text] = False
                        else:
                            matched[sent.text] = True

        #Method 2: fuzzywuzzy on the ncs + ents

        doc1 = nlp1(s)
        vals1 = list(matchSet)
        #vals1.extend(allMatches)
        sentences2=[]
        for sent in doc1.sents:
            vals2 = list(x.text for x in sent.ents)
            #vals2.extend(y.text for y in sent.noun_chunks)
            for val1 in vals1:
                for val2 in vals2:
                    if partial_ratio(val1,val2) > PARTIAL_TOLERANCE:
                        if sent.text.encode("ascii", errors="ignore").decode() == sent.text and sent.text.replace("\n", "") not in sentences2:
                            sentences2.append(sent.text.replace("\n", ""))


    pprint(sentences)
    print("----------------------------------------")
    pprint(sentences2)

    """with open('out1.txt','w') as rout:
                rout.write("\n".join(sentences))
        with open('out2.txt','w') as rout:
            rout.write("\n".join(sentences2))
        """


    return ""
    #Then to look through numbered args, returning the sents in each,


def main():
    text = " A woman comes to Mel's shop to sell antiques from a house she's moving from after the death of her daughter (from an illness). Melinda very quickly discovers that the house is haunted by violent spirits after Ned gets hurt there-which doesn't go down well with Delia- and she realizes that the ghost of the little girl (Cassidy) is being trapped there."
    doc = nlp(text)
    print(doc.sents)

"""
    while True:
        inp = input('Enter verb: ')
        sn = wn.synsets(inp)
        if len(sn):
            for s in sn:
                for lemma in wn.synset(s.name()).lemmas():
                    print(" | ".join(lemma.frame_strings()))
        else:
            print("None found")

    even = wn.synset('largely.r.01')
    unduly = wn.synset('vastly.r.01')

    vset = set()
    for synset in wn.all_synsets('r'): #adverb = r
        vset.add(synset.name().partition(".")[0])

    yes=set()
    with open('out.txt','w') as outfile:
        outfile.write("\n".join(list(vset)))
"""
main()