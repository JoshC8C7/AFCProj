from graphviz import Digraph
import matplotlib.pyplot as plt
from pprint import pprint
from graphviz import Source
import nltk

def argID(argV):
    return (str(argV.start) +"X"+ str(argV.end)+argV.text)

def getEdgeStyle(label):
    if label in []:
        return 'dotted'
    else:
        return 'solid'


#Method to handle coreferences
#If there is a registered coreference (as set under doc._.corefs) then check if its within the current span/argument,
#if there is one then return that (i.e. return 'Bob' rather than 'he'), otherwise just return []
def getCorefs(span):
    matches=[]
    for corefK, corefV in span.doc._.DCorefs.items():
        #is coref within span?
        if corefK.start >= span.start and corefK.end <=span.end:
            if not any(tok.pos_ == "PRON" for tok in corefV):
                matches.append((corefK,corefV))
    return matches

class TLClaim:

    def __init__(self, docIn,subclaims):
        self.doc = docIn

        # List of all claims made within.
        self.subclaims = subclaims



    def printTL(self):
        print("Text: ",self.doc.text)
        for sc in self.subclaims:
            sc.printCl()
        #[to_nltk_tree(sent.root).pretty_print() for sent in self.doc.sents]
        print("/////////////////////////////////////////////////////////////")

    #Takes subclaims and outputs graph relating their spans.
    def generateCG(self,doc):

        G = Digraph(format='pdf')
        argSet= set()
        verbSet = set()
        for claim in self.subclaims:
            root=claim.args['V']
            G.node(argID(root), root.text)
            for argK, argV in claim.args.items():
                if argK != 'V':
                    G.node(argID(argV), argV.text + "/" + str(argV.ents))
                    G.edge(argID(argV), argID(root), label=argK, style=getEdgeStyle(argK))
                    argSet.add(argV)  # argV = arg value, not verb.
                    for coref in argV._.SCorefs:
                        if coref[1] != argV:
                            G.node(argID(coref[1]), coref[1].text + "/" + str(coref[1].ents))
                            G.edge(argID(argV), argID(coref[1]), color='green', label=coref[0].text)

                else:
                    verbSet.add(argV)

        for argV in verbSet:
            shortestSpan=doc[:]
            for parent in argSet:
                if argV.start >= parent.start and argV.end <= parent.end and (parent.end-parent.start) < (shortestSpan.end-shortestSpan.start):
                    shortestSpan=parent
            G.edge(argID(argV),argID(shortestSpan),color='violet')


        G.save(filename=(str(hash(self.doc))))
        self.printTL()
        return

class Claim:
    def __init__(self,docIn,args,uvi):
        #Claim has Verb and a range of arguments.
        #Also a UVI resolution, and any entities found within.

        self.doc=docIn #spacy Doc - just points to parent TLClaim's doc.

        #All spacy spans pointing to the portion in question.
        self.args=args

        #Resolved UVI type - tuple of (Span in question, UVI value)
        self.uvi = uvi

    def resolveUvi(self):
        #take self.doc and resolve to find a uvi for the verb self.v.
        return None

    def printCl(self):
        print("VERB: ", self.uvi[0], " -> ", self.uvi[1])
        print("ARGS: ", end='')
        for ik, iv in self.args.items():
            if ik != 'V':
                print(ik,": ",iv.start,"->",iv.end,iv,"//",iv.ents,end='')
                corefs= iv._.SCorefs
                if corefs != []:
                    print(" // Corefs: ",corefs,"    ",type(corefs[0][0]),end='')

            print("")
        return