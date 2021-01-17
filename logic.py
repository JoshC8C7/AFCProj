from nltk.sem import Expression as expr
from nltk.inference.resolution import *
from string import ascii_lowercase

class KnowledgeBase():

    prover = None
    core = ['ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4', 'ARG5']
    other = ['SKIP', 'AND','IF','OR','RxARG0','RxARG1']

    def __init__(self, claimIn, roots,argBase):
        self.claimG = claimIn
        self.roots = roots
        self.argBaseC = argBase
        self.kb = []
        self.freeVariableCounter = 0
        self.searchTerms=[]
        self.graph2rules()

    def prepSearch(self):
        queries=[]
        if self.searchTerms == []:
            for root in self.roots:
                self.searchTerms.append((root,list(x[0] for x in self.claimG.in_edges(nbunch=root))))
        for term in self.searchTerms:
            spanList= sorted(list(self.argBaseC[arg].span for arg in term[1] + [term[0]]),key=lambda x: x.start)
            spanListCoref=[]
            for span in spanList:
                if span._.SCorefs != []:
                    nspan = list(tok for tok in span)
                    for tok in span._.SCorefs[0][0]:
                        nspan.remove(tok)
                    for tok in span._.SCorefs[0][1]:
                        nspan.append(tok)
                    nspan = sorted(nspan,key= lambda tok: tok.i)
                    for n in nspan:
                        spanListCoref.append(n)
                else:
                    for n in span:
                        spanListCoref.append(n)
            queries.append(" ".join(x.text for x in spanListCoref))
        return queries



    #Determines whether edge leads to a 'core' argument (i.e. a named one, and/or one that is not a leaf), or if
    #leads to a modifier (ARGM-) or a leaf argument ('other').
    def modOrCore(self,edge):
        if len(self.claimG.in_edges(nbunch=edge[0])) > 0:
            return 'coreInternal'
        if edge[2].get('label','') in self.core:
            return 'core'
        if edge[2].get('label','SKIP') not in self.core + self.other:
            if edge[2].get('label','SKIP') == 'ARGM-NEG':
                return 'neg'
            else:
                return 'mod'
        else:
            return 'other'

    #Generates a unique free variable for the knowledge base rules to be instantiated with.
    def getFreeVar(self):
        modifier = self.freeVariableCounter // 26
        retVal = 'a'*modifier + ascii_lowercase[self.freeVariableCounter]
        self.freeVariableCounter+=1
        return retVal

    def addToKb(self,text):
        self.kb.append(expr.fromstring(text))
        print("adding to kb ",text)
        return


    def graph2rules(self):
        #Keep track of seen nodes as to avoid cycles.
        seen = set()
        #Starting at 'root' verb(s), it being fulfilled means it implies argF(root) - conjuncted with any other root verbs in the subgraph. 'make(IG_report, clear_that) -> argF(root)'
        #To check if its fulfilled, check all in-edges.
        #1. The edge is to a simple arg leaf - it then becomes part of the parent verb i.e. sells(Tesco,____)
        #2. The edge is to an arg that is established by a tree - this is placed as fulfilling the correct argument e.g.-> sells(a1,makes_clear_that_impeachment))
        #2b. Case 2 applies to multiple edges - multiple gaps left e.g. sells(x,y).
        #3. Multiple verbs required to establish 1 arg (i.e. multiple purple edges), the below verb is implied by a conjunction of parent verbs. 'when(launch(fbi, investigation),tuesday) & sell(ducks,children) -> make(x, clear_that) '
        #FOR MODIFIERS - form a new term wrapping the verb in them e.g. starting with launch(fbi, investigation) if it happened on tuesday, we add '& when(launch(fbi,investigation), tuesday))'
        #todo cyan and red nodes?
        if len(self.roots) ==0:
            print("No roots")
            return
        #Create the root implication as the conjunction of the verbs that feed into the root.
        rootImpl = self.conjEstablish(self.roots,seen) + ' -> argF(root)'
        self.addToKb(rootImpl)
        return

    #For multiple verbs feeding one argument, the argument is implied by the conjunction of the subtrees rooted at the verbs
    #So to establish an argument which has incoming edges, we must establish all incoming edges.
    def conjEstablish(self,rootsIn,seen):
        filteredRoots=(y for y in rootsIn if len(self.claimG.in_edges(nbunch=y)) > 0)
        return " & ".join(list((self.establishRule(x,seen) for x in filteredRoots)))

    #Take a node and establish it as a predicate function, with its arguments being the verb/nodes arguments.
    def establishRule(self,root,seen):
        seen.add(root)
        argList = []
        modifiers = []
        incomingEdges = sorted(self.claimG.in_edges(nbunch=root, data=True), key=lambda x: x[2].get("label","Z")) #Sort to ensure Arg0 is processed first.
        for edge in incomingEdges:
            #Find the core args to create the predicate
            if edge[2].get('style','') != 'dotted' and self.modOrCore(edge) in ['coreInternal','core']:
                argList.append(edge[0])
        #Form the predicate - have to do this now so we can add modifiers on the next pass of the edges.
        predicate = (root) + '(' + ','.join(argList) + ')'

        if all(self.modOrCore(edge) != 'coreInternal' for edge in incomingEdges) and len(argList) > 1:
            self.searchTerms.append((root,argList))

        #Now check for any non-leaf entries or modifiers
        count=0
        for edge in incomingEdges:
            if edge[2].get('style','') == 'dotted':
                continue

            # if not a leaf argument (i.e. has incoming edges:)
            if len(self.claimG.in_edges(nbunch=edge[0])) > 0 and edge[0] not in seen:
                upVal = self.conjEstablish(list(x[0] for x in (self.claimG.in_edges(nbunch=edge[0]))),seen)

                #Fill in all bar the (count)th arguments with free variables:
                miniArgList = []
                freeVar = self.getFreeVar()
                for i in range(0,len(incomingEdges)):
                    if i == count:
                        miniArgList.append((edge[0]))
                    else:
                        miniArgList.append(freeVar + str(i))
                impliedArg = (root) + '(' + ",".join(miniArgList)+")"
                self.addToKb(upVal + ' -> ' + impliedArg)

            elif self.modOrCore(edge) == 'neg':
                predicate = '-' + predicate

            # Else if it's a modifier
            elif self.modOrCore(edge) == 'mod':
                #print("MODIFIER ", edge[2]['label']+'('+predicate+','+str(edge[0])+')')
                modifiers.append(edge[2]['label']+'('+predicate+','+(edge[0])+')')

            #Else its a leaf, so just continue without adding any extra rules or modifier

            #Increase the count as we move to the count-th argument
            count+=1

        #Add the predicates
        for m in modifiers:
            predicate += " & " + m

        return predicate


#Side-entry method to test nltk's proof systems.
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