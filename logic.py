from nltk import Prover9Command, Prover9
from nltk.sem import Expression as expr
from nltk.inference.resolution import *
from string import ascii_lowercase, punctuation
from spacy.tokens.token import Token
import claim

def getSpanCoref(span):
    corefSet = set()
    for tok in span:
        if tok._.in_coref:
            for cf in tok._.coref_clusters:
                corefSet.add(cf)
    return list(corefSet)

p = Prover9()
p.config_prover9(binary_location='.')
class KnowledgeBase():

    prover = None
    core = ['ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4', 'ARG5']
    other = ['SKIP', 'AND','IF','OR','RxARG0','RxARG1']
    c = expr.fromstring('argF(root)')
    notC=expr.fromstring('-argF(root)')

    def __init__(self, claimIn, roots,tlClaim):
        self.claimG = claimIn
        self.roots = roots
        self.argBaseC = tlClaim.argBaseC
        self.kb = []
        self.freeVariableCounter = 0
        self.searchTerms=[]
        self.argFunc = tlClaim.argID
        self.kb2 = []
        self.kb2_args = {}
        self.ruleLength = 0
        self.evidenceMap = {}
        self.graph2rules()


    def getEnabledArgBaseC(self):
        filtered = {}
        for ik,iv in self.argBaseC.items():
            if iv.enabled:
                filtered[ik] = iv
        return filtered

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
                    #else:
                        #print("Dropping:",tok)
                if newNc:
                    appText = ''.join(tok.text_with_ws for tok in newNc)
                    if appText[-1] == " ":
                        appText=appText[:-1]
                    newNCs.add(appText)

        #print("ST: ", self.searchTerms, " QU:", queries)y

        #todo this is a short hack to remove searchterms that are within another one, as to reduce the horrific runtimes.
        q2=queries.copy()
        for i in queries:
            for j in queries:
                if i in j and i != j and i in q2:
                    print("rem",i," in ",j)
                    q2.remove(i)
        print("Queries",newNCs)
        return [','.join(sorted(newNCs))] + q2, ncs, entities


    #Determines whether edge leads to a 'core' argument (i.e. a named one, and/or one that is not a leaf), or if
    #leads to a modifier (ARGM-) or a leaf argument ('other').
    def modOrCore(self,edge):
        if len(list(x for x in self.claimG.in_edges(nbunch=edge[0],data=True) if x[2].get('style','') != 'dotted')) and 'ARGM' not in edge[2].get('label',''):
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
        modifier = self.freeVariableCounter
        retVal = 'u'+str(modifier)
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
        self.ruleLength = len(self.kb)
        #print("KBRULES ", self.kb)
        #self.addToKb('argF(root) ->'+ self.conjEstablish(self.roots,seen))
        return


    def temporalReasoning(self,span, modID):
        #print("Relative date:", span.doc._.rootDate)
        # Temporal reasoning imports
        from dateutil.relativedelta import relativedelta, MO

        from datetime import timedelta
        import dateutil.parser as parser

        if len(span.ents) + len(list(span.noun_chunks)) == 0 and span[0].tag_ == 'RB':
            #print("OUT:", span, "  ", span[0].tag_, span[0].dep_)
            return None

        date=None
        if span.doc._.rootDate is not None:
            try:
                date = parser.parse(span.doc._.rootDate)
            except ValueError:
                pass

        modType='TMP'
        if any(x in span.lower_ for x in ['before', 'until', 'prior to']):
            modType = 'BEFORE'
        elif any(x in span.lower_ for x in ['during','whilst','when','over']):
            modType = 'DURING'
        elif any(x in span.lower_ for x in ['after','since','subsequent']):
            modType = 'AFTER'

        #Check if arg is an internal node:
        if self.claimG.in_edges(nbunch=modID):
            return (modType,modID)

        #Else do temporal reasoning:
        if date is None:
            for ent in span.ents:
                if ent.label_ in ["TIME","DATE"]:

                    try:
                        date = parser.parse(ent.text, dayfirst=True) #todo this needs to return a span so the argId works - it should just take the id of what its replacing or something
                    except ValueError:
                        date = ent
                    #print(ent.text, "/ ", ent.label_, "/",resolvedDate)
        else:
            raise NotImplementedError
        return
        return (modType, self.argFunc(date))





    #For multiple verbs feeding one argument, the argument is implied by the conjunction of the subtrees rooted at the verbs
    #So to establish an argument which has incoming edges, we must establish all incoming edges.
    def conjEstablish(self,rootsIn,seen):
        filteredRoots=(y for y in rootsIn if len(self.claimG.in_edges(nbunch=y)) > 0)
        k=list(filteredRoots)
        inc = list((self.establishRule(x,seen) for x in k))
        self.kb2.extend(inc)
        return " & ".join(inc)

    #Take a node and establish it as a predicate function, with its arguments being the verb (node)'s arguments.
    def establishRule(self,root,seen):
        self.argBaseC[root].enableNode()
        seen.add(root)
        predNeg = False
        argList = []
        modifiers = []
        #Find all edges coming into the root verb.
        incomingEdges = sorted(self.claimG.in_edges(nbunch=root, data=True), key=lambda x: x[2].get("label","Z")) #Sort to ensure Arg0 is processed first.
        for edge in incomingEdges:
            #Find the core args to create the predicate. Modifiers are not permitted in the predicate at this point.
            if edge[2].get('style','') != 'dotted' and self.modOrCore(edge) in ['coreInternal','core']:
                argList.append(edge[0])

        #if all(self.modOrCore(edge) != 'coreInternal' for edge in incomingEdges) and len(argList) > 1:
        if len(argList) > 1:
            #todo there's some duplication of search terms going on here - see John Bolton
            self.searchTerms.append((root,argList))
        """        else:
            print("Skipsies",root,argList)"""

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
                impliedArg = (root)+str(len(miniArgList)) + '(' + ",".join(miniArgList)+")"
                self.kb2.append(impliedArg)
                if upVal:
                    self.addToKb(upVal + ' -> ' + impliedArg)

            elif self.modOrCore(edge) == 'neg':
                predNeg = True

            # Else if it's a modifier
            elif self.modOrCore(edge) == 'mod':
                #print("mod", edge)
                modType = edge[2]['label'].replace("ARGMx","")
                modValID = edge[0]
                modVal = self.argBaseC[modValID].span

                """if modType == 'TMP':
                    if 'never' in modVal.lower_:
                        predNeg = True
                    else:
                        #send to temporal reasoning
                        #print("Temporal: ",modVal, list(modVal.noun_chunks), modVal.ents)
                        ret = self.temporalReasoning(modVal,modValID)
                        print("RET",ret)
                        if ret is not None:
                            modifiers.append(ret)
                            modType = ret[0]
                            modValID = ret[1]"""
                if modType in ['TMP']+['MOD','ADV','PRP', 'CAU', 'LOC']:
                    if 'never' in modVal.lower_:
                        predNeg = True
                    else:
                        modifiers.append((modType, modValID))
                        self.argBaseC[modValID].enableNode()
                elif modType in ['DIR', 'PRD','MNR']:
                    if not(modType == 'MNR' and all(tok.tag_ not in claim.grammaticalTAG for tok in modVal)):
                        argList.append(modValID)


                #print("MODIFIER ", modType+'('+str(argList)+','+str(modValID)+')')

                #If the modifier has incoming edges from verbs that root subtrees (i.e. the modifier contains 1 or more verbs):
                if len(self.claimG.in_edges(nbunch=edge[0])) > 0 and edge[0] not in seen:
                    upVal = self.conjEstablish(list(x[0] for x in (self.claimG.in_edges(nbunch=edge[0]))),seen)  # Conjestablish over all incoming violet edges (although usually is just 1)
                    impliedArg = modType + '(' +self.getFreeVar()+','+ modValID + ')' #The free value here represents the predicate.
                    self.addToKb(upVal + ' -> ' + impliedArg)


            #Else its a leaf, so just continue without adding any extra rules or modifier

            #Increase the count as we move to the count-th argument (i.e. exclude Modifiers -
            # theoretically this shouldn't matter as count falls out of use after numbered arguments, which come first)
            if 'ARGM' not in edge[2].get('label',''):
                count+=1

        # Form the predicate - have to do this now so we can add modifiers on the next pass of the edges.
        predicate = root + str(len(argList)) + '(' + ','.join(argList) + ')'

        for arg in argList:
            self.argBaseC[arg].enableNode()

        if predNeg:
            print("predneg")
            predicate = '-'+predicate

        #Add the predicates
        oldPred = predicate
        self.kb2_args[oldPred] = []
        for m in modifiers:
            modifierText = m[0]+oldPred.translate(str.maketrans('', '', punctuation)) + '(' + m[1] + ')'
            self.kb2_args[oldPred].append(modifierText)
            predicate += " & " + modifierText

        return predicate


    def prove(self,system='res'):
        print("proving....")
        if system == 'res':
            resProv = ResolutionProverCommand
        else:
            resProv = Prover9Command
        rpc = resProv(goal=self.c,assumptions=self.kb)
        p1 = rpc.prove(verbose=True)
        if p1:
            prf = rpc.proof().replace(" ", "").split("\n")
            prfParsed = []
            for x in prf:
                if '{}' in x:
                    try:
                        y = x.split('}')[1].replace('(','').replace(')','').split(',')
                        prfParsed.append((int(y[0]),int(y[1])))
                    except:
                        continue #in the rare case that argF{} is read as an input.
                elif x:
                    y=x.split('{')[1].split('}')
                    if y[1] == 'A':
                        prfParsed.append(y[0])
                    else:
                        ysplit = y[1].replace(')','').replace('(','').split(',')
                        prfParsed.append((int(ysplit[0]),int(ysplit[1])))

            #print(self.ruleLength)
            """for ind, a in enumerate(prfParsed):
                if ind <= self.ruleLength:
                    print("(Rule) ",end='')
                print(str(ind+1), a)"""
            path = backtracker(prfParsed,self.ruleLength)
            for index, i in enumerate(path):
                if index > 0 and index < len(path) - 1: print("-->--",end="")
                if type(i) is not tuple:
                    print(self.evidenceMap[i], " @ ", self.evidenceMap[i]._.url,end='')
                #else: print(i)
            print()
        return p1
        #print(rpc.proof(simplify=True))

def backtracker(prf, rl):
    from collections import deque
    path=[]
    queue = deque(prf[-1])
    while len(queue):
        k=queue.popleft()-1
        step = prf[k]
        if k <= rl:
            print("",end='')
            #print("RULE",end='')
        else:
            path.append(step)
        #print('STEP: ', step)
        if type(step) is tuple:
            queue.extend(step)
    path.reverse()
    return path

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

    #kb.append(expr.fromstring('launch(fbi, investigation) & when(launch(fbi, investigation),tuesday) & how(launch(fbi,investigation), reluctantly) & sell(ducks,children) -> make(x, clear_that) ')) #Having both parent args -> right child arg implied
    #kb.append(expr.fromstring('make(IG_report, clear_that) -> argF(root)')) #Having both parent args -> right (and only) child implied.

    c = expr.fromstring('argF(root)')
    """z1 = expr.fromstring('6X7Xasked(0X4XTheDemocratcontrolledHouse,9X11Xtotestify,7X9XJohnBolton)')
    z2 = expr.fromstring('6X7Xasked(0X4XTheDemocratcontrolledHouse,9X11Xtotestify,7X9XJohnBolton) -> argF(root)')
    kb.append(z1)
    kb.append(z2)
    print(type(z1))
    print(ResolutionProverCommand(c, kb).prove(verbose=False))"""


    #Then to satisfy...
    #kb.append(expr.fromstring('when(launch(fbi, investigation), tuesday)')) #As this is a leaf and one of the arguments doesn't involve further nodes/a subtree, both args are offered here #1.

    #rules
    kb.append(expr.fromstring('sells(cake,bob) & wants(cake,bob) -> argF(root)'))
    kb.append(expr.fromstring('has(cake,shop) & open(shop) -> sells(cake,x)'))

    #evidence
    kb.append(expr.fromstring('open(shop)'))
    kb.append(expr.fromstring('wants(cake,bob)'))
    kb.append(expr.fromstring('has(cake,shop)'))



    #Code to run it....
    rpc=ResolutionProverCommand(c,kb)
    rpc.prove(verbose=True)


    #todo when the proof fails, we get the remaining terms to focus on proving - then see if we can unify that even partially with something to create the 'new front', need to see what else is known but not used.
    #Notice how X(lobsters) is an entry above but isn't used. We know this because 5 doesn't make a further appearence on the right - this is how you can extract the new front.
    #Can consider moving away from NLTK - not any real reason to be bound to it.


    """kb=[]
    kb.append(expr.fromstring('a -> True'))
    c = expr.fromstring('b')
    print(ResolutionProverCommand(c, kb).prove(verbose=True))"""