from nltk.sem import Expression as expr
from nltk.inference.resolution import *
import string

class KnowledgeBase():

    prover = None
    core = ['ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4', 'ARG5']
    other = ['SKIP', 'AND','IF','OR','RxARG0','RxARG1']

    def __init__(self, claimIn, roots,argBase):
        self.claimG = claimIn
        self.roots = roots
        self.argBase = argBase
        self.kb = []
        self.freeVariableCounter = 0


        self.graph2rules()

    def modOrCore(self,edge):
        if edge[2].get('label','') in self.core:
            return 'core'
        if len(self.claimG.in_edges(nbunch=edge[0])) > 0:
            return 'core'
        if edge[2].get('label','SKIP') not in self.core + self.other:
            return 'mod'
        else:
            return 'other'

    #Sanitizes string for rule instantiation, converting from graph-arg to KB-arg
    def formatArg(self,arg):

        #Retrieve spaCy Span from encoded graph node
        argSpan = self.argBase[arg]

        if len(argSpan._.SCorefs) > 0:
            # push in coreferences if they exist
            tokMap = {}
            for coref in argSpan._.SCorefs:
                for tok in argSpan:
                    if tok.i == coref[0].start:
                        tokMap[tok] = ''.join(x.text for x in coref[1])
                    elif tok.i in range(coref[0].start,coref[0].end):
                        tokMap[tok] = ''
            newArgStr = ''
            for tok in argSpan:
                if tok in tokMap:
                    newArgStr += tokMap[tok]
                else:
                    newArgStr += tok.text
            #as retval is just a string here, have to generate its KB-arg manually.
            retVal = (str(argSpan.start) + "X" + str(argSpan.end) + newArgStr.replace("\"", ""))
        else:
            retVal = arg

        retVal = ''.join(filter(str.isalnum, str(retVal))).replace(' ', '')
        return 'k' + retVal


    def getFreeVar(self):
        modifier = self.freeVariableCounter // 26
        retVal = 'a'*modifier + string.ascii_lowercase[self.freeVariableCounter]
        self.freeVariableCounter+=1
        return retVal

    def addToKb(self,text):
        self.kb.append(expr.fromstring(text))
        print("adding to kb ",text)
        return


    def graph2rules(self):
        seen = set()
        #Starting at 'root' verb(s), it being fulfilled means it implies argF(root) - conjuncted with any other root verbs in the subgraph. 'make(IG_report, clear_that) -> argF(root)'
        #To check if its fulfilled, check all in-edges.
        #1. The edge is to a simple arg leaf - it then becomes part of the parent verb i.e. sells(Tesco,____)
        #2. The edge is to an arg that is established by a tree - this is placed as fulfilling the correct argument e.g.-> sells(x,parentTree(something,__))
        #2b. Case 2 applies to multiple edges - multiple gaps left e.g. sells(x,y).
        #3. Multiple verbs required to establish 1 arg (i.e. multiple purple edges), the below verb is implied by a conjunction of parent verbs. 'when(launch(fbi, investigation),tuesday) & sell(ducks,children) -> make(x, clear_that) '
        #When determining what to term, post to the coref checker to sub in any required coreferences.
        #FOR MODIFIERS - form a new term wrapping the verb in them e.g. starting with launch(fbi, investigation) if it happened on tuesday, we add '& when(launch(fbi,investigation), tuesday))'
        #todo cyan and red nodes?
        if len(self.roots) ==0:
            print("No roots")
            return
        rootImpl = self.conjEstablish(self.roots,seen) + ' -> argF(root)'
        self.addToKb(rootImpl)
        #pprint(self.kb)
        return

    def conjEstablish(self,rootsIn,seen):
        filteredRoots=(y for y in rootsIn if len(self.claimG.in_edges(nbunch=y)) > 0)
        return " & ".join(list((self.establishRule(x,seen) for x in filteredRoots)))

    def establishRule(self,root,seen):
        seen.add(root)
        argList = []
        modifiers = []
        incomingEdges = sorted(self.claimG.in_edges(nbunch=root, data=True), key=lambda x: x[2].get("label","Z")) #Sort to ensure Arg0 is processed first.
        for edge in incomingEdges:
            #print("EDGE",edge)
            #Find the core args to create the predicate
            if edge[2].get('style','') != 'dotted' and self.modOrCore(edge) == 'core':
                #Add the sanitised (for hashing purposes) argument at the end of the edge
                argList.append(self.formatArg(edge[0]))
        #Form the predicate - have to do this now so we can add modifiers on the next pass of the edges.
        predicate = self.formatArg(root) + '(' + ','.join(argList) + ')'

        #Now check for any non-leaf entries or modifiers
        count=0
        for edge in incomingEdges:

            # it isn't a leaf argument
            if edge[2].get('style','') != 'dotted' and len(self.claimG.in_edges(nbunch=edge[0])) > 0 and edge[0] not in seen:
                upVal = self.conjEstablish(list(x[0] for x in (self.claimG.in_edges(nbunch=edge[0]))),seen)

                #Fill in all bar the (count)th arguments with free variables:
                miniArgList = []
                freeVar = self.getFreeVar()
                for i in range(0,len(incomingEdges)):
                    if i == count:
                        miniArgList.append(self.formatArg(edge[0]))
                    else:
                        miniArgList.append(freeVar + str(i))
                impliedArg = self.formatArg(root) + '(' + ",".join(miniArgList)+")"
                self.addToKb(upVal + ' -> ' + impliedArg)

            # Else if it's a modifier
            elif edge[2].get('style','') != 'dotted' and self.modOrCore(edge) == 'mod':
                #print("MODIFIER ", edge[2]['label']+'('+predicate+','+str(edge[0])+')')
                modifiers.append(edge[2]['label']+'('+predicate+','+self.formatArg(edge[0])+')')
            #else its a leaf, so just continue without adding any extra rules or modifier

            #Increase the count as we move to the count-th argument
            count+=1

        for m in modifiers:
            predicate += " & " + m

        return predicate



if __name__ == '__main__':
    print("main")
    ns = expr.fromstring('launch(t,u) & make(x,y) -> Root')
    #print(type(expr.fromstring('walk(P)', type_check=True).argument))
    #'walk(P)' #<class 'nltk.sem.logic.FunctionVariableExpression'>
    #'walk(pablo)' #<class 'nltk.sem.logic.ConstantExpression'>
    #'walk(a)'-> #<class 'nltk.sem.logic.IndividualVariableExpression'>
    kb = []

    #Examples: argF is that a certain nodes prerequesites are satisfied.

    #Rules:

    kb.append(expr.fromstring('launch(fbi, investigation) & when(launch(fbi, investigation),tuesday) & how(launch(fbi,investigation), reluctantly) & sell(ducks,children) -> make(x, clear_that) ')) #Having both parent args -> right child arg implied
    kb.append(expr.fromstring('make(IG_report, clear_that) -> argF(root)')) #Having both parent args -> right (and only) child implied.
    c = expr.fromstring('argF(root)')

    #Then to satisfy...
    kb.append(expr.fromstring('when(launch(fbi, investigation), tuesday)')) #As this is a leaf and one of the arguments doesn't involve further nodes/a subtree, both args are offered here #1.
    kb.append(expr.fromstring('how(launch(fbi, investigation), reluctantly)'))
    kb.append(expr.fromstring('sell(ducks, children)')) #2
    kb.append(expr.fromstring('X(lobsters)'))
    kb.append(expr.fromstring('make(IG_report,y)')) #Conversely, the 2nd arg here is a subtree, so will have a gap to be filled.

    #Code to run it....
    rpc=ResolutionProverCommand(c,kb)
    print(rpc.prove(verbose=True))
    print(rpc.proof())


    #todo when the proof fails, we get the remaining terms to focus on proving - then see if we can unify that even partially with something to create the 'new front', need to see what else is known but not used.
    #Notice how X(lobsters) is an entry above but isn't used. We know this because 5 doesn't make a further appearence on the right - this is how you can extract the new front.
    #Can consider moving away from NLTK - not any real reason to be bound to it.


    """kb=[]
    kb.append(expr.fromstring('a -> True'))
    c = expr.fromstring('b')
    print(ResolutionProverCommand(c, kb).prove(verbose=True))"""


