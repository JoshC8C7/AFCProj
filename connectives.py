import spacy
from spacy import displacy
import pandas as pd
import os
from pathlib import Path
from spacy.tokens import Span, Doc


class Connective:
    #For IF, start = condition and end = consequence.
    #todo convert types to enum
    connType = None
    start = None
    end = None
    note = None
    colours={'ARG0':'black','IF':'crimson','AND':'cyan','OR':'cyan'}

    def __init__(self,connType,start,end,note=None):
        self.connType=connType
        self.start=start
        self.end=end
        self.note=note

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "Type: "+self.connType+ " /Start: "+ self.start.text+ " /End: "+self.end.text+" /Note: "+self.note

#SpaCy pipeline components take doc and return doc.
def extractConnectives(doc):
    subjects=['csubj','subjpass','nsubj','nsubjpass']
    effect = None
    edges = []

    # Edge #1 - No verb + conj
    for nc in doc.noun_chunks:
        if nc.root.head.pos_ == 'NOUN' and nc.root.dep_ == 'conj':
            parent = nc.root.head
            while parent.dep_ == 'conj':
                parent = parent.head
            if parent.head.pos_ == "VERB":
                print(" NEW EDGE: ", nc.root, "-ARG0>", parent.head)
                edges.append(Connective('ARG0',nc,Span(doc,parent.head.i,parent.head.i+1),'ARG0-E1'))

    for tok in doc:
        # Edge #2 -> and
        if tok.pos_ == "VERB" and tok.dep_ == "conj" and tok.head.pos_=="VERB" and 'or' not in doc[min(tok.i,tok.head.i):max(tok.i,tok.head.i)+1].text:
            print("NEW EDGE: ", tok, "--and-->", tok.head)
            edges.append(Connective('AND',Span(doc,tok.i,tok.i+1),Span(doc,tok.head.i,tok.head.i+1),'AND-E2'))
            if not any(child.dep_ in subjects for child in tok.children):
                for pchild in tok.head.children:
                    if pchild != tok and pchild.dep_ in subjects:
                        print("ADOPT SUBJ", pchild, "-ARG0->", tok)
                        edges.append(Connective('ARG0',Span(doc,pchild.i,pchild.i+1),Span(doc,tok.i,tok.i+1),'ARG0-E2/ADOPT SUBJ'))

        #Edge #3 -> if
        if tok.lower_ == 'if' and tok.head.pos_ in ["VERB","AUX"]:
            condition = list(tok.head.subtree)
            if tok.head.dep_ == 'advcl': #Issue is here - this isn't triggering.
                effect = list(x for x in tok.head.head.subtree if x not in condition)
            if effect is not None:
                condition.remove(tok)
                print("NEW EDGE: ", "IF: ", condition, "THEN:",effect)
                print(Span(doc,condition[0].i,condition[-1].i+1))
                print(Span(doc,effect[0].i,effect[-1].i+1))
                edges.append(Connective('IF',Span(doc,condition[0].i,condition[-1].i+1),Span(doc,effect[0].i,effect[-1].i+1),'IF source THEN dest-E3'))


        #Edge #4 -> or
        if tok.lower_ == 'or':
            for child in tok.head.children:
                if child.dep_ == 'conj':
                    print("NEW EDGE", tok.head, '-or->', child)
                    edges.append(Connective('OR',Span(doc,tok.head.i,tok.head.i+1),Span(doc,child.i,child.i+1),'OR-E4'))
                    break

    """svg = displacy.render(doc, style="dep")
    output_path = Path(tok.head.text + ".svg")
    output_path.open("w", encoding="utf-8").write(svg)"""
    doc._.ConnectiveEdges = edges
    return doc


def test():
    df = pd.read_table(os.path.join("data", "Politihop", "Politihop_train.tsv"), sep='\t')
    statementSet = set()
    nlp=spacy.load('en_core_web_lg')
    Doc.set_extension("ConnectiveEdges", default=[])

    for s in df['statement']:
        while not s[0].isalpha():
            s = s[1:]
        if s.split(" ")[0].lower() == "says":
            s = (s.partition(" "))[1]
        s.replace('&amp;', 'and')
        statementSet.add(s)

    sought=['if','and','or','but','however','neither','either','although','so','with','without']
    for s in statementSet:
        doc=nlp(s)
        if any(tok.lower_ in sought for tok in doc):
            extractConnectives(doc)


#test()