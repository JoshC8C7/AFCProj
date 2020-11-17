class TLClaim:

    def __init__(self, docIn,subclaims):
        self.doc = docIn

        # List of all claims made within.
        self.subclaims = subclaims

    def printTL(self):
        print("Text: ",self.doc.text)
        for sc in self.subclaims:
            sc.printCl()

    #Takes subclaims and outputs graph relating their spans.
    def generateCG(self):
        claimGraph = 1
        return claimGraph

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
                print(ik,": ",iv,"//",iv.ents," ",end='')
        print("")
        return