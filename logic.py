from nltk.sem import Expression as expr
from nltk.inference.resolution import *
from string import ascii_lowercase
from spacy.tokens.token import Token
import claim

def getSpanCoref(span):
    corefSet = set()
    for tok in span:
        if tok._.in_coref:
            for cf in tok._.coref_clusters:
                corefSet.add(cf)
    return list(corefSet)

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
        entities=[]
        ncs=[]
        newNCs=set()
        if not self.searchTerms:
            for root in self.roots:
                self.searchTerms.append((root,list(x[0] for x in self.claimG.in_edges(nbunch=root))))
        for term in self.searchTerms:
            spanList= sorted(list(self.argBaseC[arg].span for arg in term[1] + [term[0]]),key=lambda x: x.start)
            corefSubbedSpans = []
            for span in spanList:
                entities.extend(span.ents)
                ncs.extend(span.noun_chunks)
                usedCorefs = []
                newSpan = []
                for tok in span:
                    if tok._.in_coref and tok.tag_ in ('PRP', 'PRP$'):
                        coref = tok._.coref_clusters[0]
                        if coref not in usedCorefs:
                            newSpan.extend(tok for tok in coref.main)
                            usedCorefs.append(coref)
                            entities.extend(coref.main.ents)
                        if tok.tag_ == 'PRP$':
                            newSpan.append("'s ")
                    else:
                        newSpan.append(tok)
                corefSubbedSpans.append(''.join((tok.text_with_ws if type(tok) == Token else tok) for tok in newSpan))
            queries.append(' '.join(corefSubbedSpans).replace('  ',' ').replace(" 's","'s"))

            for val in ncs+entities:
                newNc=[]
                for tok in val:
                    if tok.tag_ not in claim.grammaticalTAG + ['PRP','PRP$']:
                        newNc.append(tok)
                    else:
                        print("Dropping:",tok)
                if newNc:
                    appText = ''.join(tok.text_with_ws for tok in newNc)
                    if appText[-1] == " ":
                        appText=appText[:-1]
                    newNCs.add(appText)

        #print("ST: ", self.searchTerms, " QU:", queries)y

        return queries, newNCs


    #Determines whether edge leads to a 'core' argument (i.e. a named one, and/or one that is not a leaf), or if
    #leads to a modifier (ARGM-) or a leaf argument ('other').
    def modOrCore(self,edge):
        if len(list(x for x in self.claimG.in_edges(nbunch=edge[0],data=True) if x[2].get('style','') != 'dotted')):
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
        retVal = 'a'*modifier + ascii_lowercase[self.freeVariableCounter // 26]
        self.freeVariableCounter+=1
        return retVal

    def addToKb(self,text):
        print("adding to kb ", text)
        self.kb.append(expr.fromstring(text))
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
            #print("No roots")
            return
        #Create the root implication as the conjunction of the verbs that feed into the root.
        rootImpl = self.conjEstablish(self.roots,seen) + ' -> argF(root)'
        self.addToKb(rootImpl)
        print("//////////////////////////////")
        return


    def temporalReasoning(self,span):
        #print("Relative date:", span.doc._.rootDate)
        # Temporal reasoning imports
        from dateutil.relativedelta import relativedelta, MO

        from datetime import timedelta
        import dateutil.parser as parser

        if len(span.ents) + len(list(span.noun_chunks)) == 0 and span[0].tag_ == 'RB':
            print("OUT:", span, "  ", span[0].tag_, span[0].dep_)
            return None

        print(span.doc.text)
        date=None
        if span.doc._.rootDate is not None:
            try:
                date = parser.parse(span.doc._.rootDate)
            except ValueError:
                pass

        #todo if before - argtype is BEFORE(x, resolvedDate) rather than TMP.
        # So just set the modtype to BEFORE, x stays as the verb-being-modified, and the value is the arg text, and do
        #relative date res if its a leaf - no need to establish furhter trees, its done later

        return None





    #For multiple verbs feeding one argument, the argument is implied by the conjunction of the subtrees rooted at the verbs
    #So to establish an argument which has incoming edges, we must establish all incoming edges.
    def conjEstablish(self,rootsIn,seen):
        filteredRoots=(y for y in rootsIn if len(self.claimG.in_edges(nbunch=y)) > 0)
        k=list(filteredRoots)
        return " & ".join(list((self.establishRule(x,seen) for x in k)))

    #Take a node and establish it as a predicate function, with its arguments being the verb (node)'s arguments.
    def establishRule(self,root,seen):
        seen.add(root)
        predNeg = False
        argList = []
        modifiers = []
        #Find all edges coming into the root verb.
        incomingEdges = sorted(self.claimG.in_edges(nbunch=root, data=True), key=lambda x: x[2].get("label","Z")) #Sort to ensure Arg0 is processed first.
        for edge in incomingEdges:
            #Find the core args to create the predicate. Modifiers are not permitted in the predicate at this point.
            if edge[2].get('style','') != 'dotted' and self.modOrCore(edge) in ['coreInternal','core'] and 'ARGM' not in edge[2].get('label',''):
                argList.append(edge[0])

        if all(self.modOrCore(edge) != 'coreInternal' for edge in incomingEdges) and len(argList) > 1:
            self.searchTerms.append((root,argList))

        #Now check for any non-leaf entries or modifiers
        count=0
        for edge in incomingEdges: #Arg0, Arg1 etc for the root verb.
            if edge[2].get('style','') == 'dotted':
                continue

            # Is argx an internal node (i.e. has incoming violet/verb-subpart edges:) - NOT modifiers (although they can have incoming violet edges, this is handled later)
            #If so, need to handle the subtree rooted at it (i.e. recurse deeper)
            if len(self.claimG.in_edges(nbunch=edge[0])) > 0 and 'ARGM' not in edge[2].get('label','') and edge[0] not in seen: #Looking at an arg, we're just checking if it has at least 1 violet edge in. Modifiers are not args.
                upVal = self.conjEstablish(list(x[0] for x in (self.claimG.in_edges(nbunch=edge[0]))),seen) #Conjestablish over all incoming violet edges (although usually is just 1)

                #Fill in all bar the (count)th arguments with free variables:
                miniArgList = []
                freeVar = self.getFreeVar()
                for i in range(0,len(list(x for x in incomingEdges if 'ARGM' not in x[2].get('label','')))):
                    if i == count:
                        miniArgList.append((edge[0]))
                    else:
                        miniArgList.append(freeVar + str(i))
                impliedArg = (root) + '(' + ",".join(miniArgList)+")"
                self.addToKb(upVal + ' -> ' + impliedArg)

            elif self.modOrCore(edge) == 'neg':
                predNeg = True

            # Else if it's a modifier
            elif self.modOrCore(edge) == 'mod':
                modType = edge[2]['label'].replace("ARGMx","")
                modValID = edge[0]
                modVal = self.argBaseC[modValID].span

                if modType == 'TMP':
                    if 'never' in modVal.lower_:
                        predNeg = True
                    else:
                        #send to temporal reasoning
                        #print("Temporal: ",modVal, list(modVal.noun_chunks), modVal.ents)
                        ret = self.temporalReasoning(modVal)
                        if ret is not None:
                            modifiers.append(ret)
                elif modType in ['MOD','ADV','PRP', 'CAU', 'LOC']:
                    modifiers.append((modType, modValID))
                elif modType in ['DIR', 'PRD','MNR']:
                    if not(modType == 'MNR' and all(tok.tag_ not in claim.grammaticalTAG for tok in modVal)):
                        argList.append(modValID)

                #todo check if the modifier node roots a tree, and then do the relevant as above - make sure it implies the right modType (e.g. could be BEFORE)
                #print("MODIFIER ", modType+'('+str(argList)+','+str(modValID)+')')


            #Else its a leaf, so just continue without adding any extra rules or modifier

            #Increase the count as we move to the count-th argument (i.e. exclude Modifiers -
            # theoretically this shouldn't matter as count falls out of use after numbered arguments, which come first)
            if 'ARGM' not in edge[2].get('label',''):
                count+=1

        # Form the predicate - have to do this now so we can add modifiers on the next pass of the edges.
        predicate = root + '(' + ','.join(argList) + ')'
        if predNeg:
            predicate = '-'+predicate

        #Add the predicates
        for m in modifiers:
            modifierText = m[0] + '(' + predicate + ',' + m[1] + ')'
            predicate += " & " + modifierText

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