from graphviz import Digraph, Source
from pprint import pprint
import networkx as nx
from pydot import graph_from_dot_data
from logic import KnowledgeBase

def getEdgeStyle(label):
    if label in ['ARGM-MOD', 'AND', 'OR', 'IF']:
        return 'dotted'
    else:
        return 'solid'

def argIDGen(argV):
    debug = True
    if debug:
        #For debug, use an arg generating function that yields closer to the existing string
        k=(str(argV.start) + "X" + str(argV.end) +"X" +(argV.text.replace("\"", "").replace(":",""))) #Sanitize for graph
        j = ''.join(filter(str.isalnum, str(k))).replace(' ', '') #Pre-emptively sanitize for KB also
        return j
    else: #Otherwise just use a hash
        return (str(argV.start) + "X" + str(argV.end) +"X" +str(hash(argV.text)).replace("-","a"))

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

class argNode():

    #Doc is not stored explicitly as it will not be required from this point. Still obtainable from getDoc for debug.
    def __init__(self,ID,doc,span):
        self.ID = ID
        self.uvi = doc._.Uvis.get(span, None)
        self.span = span
        return

    def getUviOutput(self):
        if self.uvi is None: return "No Uvi Found"
        else: return str(self.uvi)

    def getDoc(self):
        return self.span.doc


class TLClaim:

    def __init__(self, docIn,OIEsubclaims):
        self.argBaseC = {}
        self.doc = docIn

        # Takes a list of OIE subclaims, note that these are not the same as subclaims obtained from the graph.
        #To convert, need to form the abstract meaning representation graph:
        self.graph = self.generateCG(OIEsubclaims, True)

        #...then extract the subclaims from the graph.
        self.subclaims = self.extractSubclaims()


    def printTL(self):
        print("Text: ",self.doc.text)
        if len(self.doc._.ConnectiveEdges) > 0:
            print("Connectives:")
            pprint(self.doc._.ConnectiveEdges)
        for sc in self.subclaims:
            sc.ClaimPrint()
        print("/////////////////////////////////////////////////////////////")

    def argID(self,argV):
        argID = argIDGen(argV)

        #Maintain a mapping of argID's being added to the graph & their respective properties.
        if argID not in self.argBaseC:
            self.argBaseC[argID] = argNode(argID, self.doc, argV)

        return argID

    #Takes subclaims and outputs graph relating their spans.
    def generateCG(self,OIEsubclaims, output=False):
        G = Digraph(strict=True,format='pdf')
        argSet= set()
        verbSet = set()
        corefNodes, corefEdges = [], []
        G.node(self.argID(self.doc[:]), self.doc.text)
        for claim in OIEsubclaims:

            root=claim['V']
            check = ''.join(filter(str.isalnum, str(root))).replace(' ', '')
            if check =="": #Sometimes some nonsense can be attributed as a verb by oie.
                continue
            G.node(self.argID(root), root.text + '/' + self.argBaseC[self.argID(root)].getUviOutput())

            for argK, argV in claim.items():
                if argK != 'V':
                    G.node(self.argID(argV), argV.text + "/" + str(argV.ents))
                    G.edge(self.argID(argV), self.argID(root), label=argK.replace('-', 'x'), style=getEdgeStyle(argK))
                    # Replace any '-' with 'x' as '-' is a keyword for networkx, but is output by allennlp
                    argSet.add(argV)  # argV = arg value, not verb.

                    # If coref edges are left on the networkx
                    #graphs then they interfere with splitting into subclaims as the edges becomes bridges. The
                    #coreferences themselves are not lost as they're properties of the doc/spans. They are useful for
                    #illustratory and debugging purposes, and so can be output when requested with output=True
                    #This is deemed sound to omit from the networkx graph as a coreference does not result in two claims
                    #being co-dependent e.g. 'My son is 11. He likes to eat cake.' - the coreference bridges the two
                    #otherwise separate components when there should be no co-dependence implied. Both are about the
                    #son, but there is not an iff relation between them.
                    if output:
                        for coref in argV._.SCorefs:
                            if coref[1] != argV:
                                corefNodes.append((self.argID(coref[1]), coref[1].text + "/" + str(coref[1].ents)))
                                corefEdges.append((self.argID(argV), self.argID(coref[1]), coref[0].text))

                else:
                    verbSet.add(argV)

        for edge in self.doc._.ConnectiveEdges:
            G.node(self.argID(edge.start), edge.start.text)
            G.node(self.argID(edge.end), edge.end.text)
            G.edge(self.argID(edge.start), self.argID(edge.end), color=edge.colours[edge.connType], label=edge.connType, style=getEdgeStyle(edge.connType))

            if edge.connType == 'IF':
                #todo fix these 3 lines - the 'TV' including example ends up with having not a verb as the parent of the root.
                #argSet.add(edge.start)
                #argSet.add(edge.end)
                #verbSet.add(edge.end)
                None

            elif edge.connType == 'OR':
                verbSet.add(edge.start)
                verbSet.add(edge.end)

        for argV in verbSet:
            shortestSpan=self.doc[:]
            for parent in argSet:
                if argV != parent and argV.start >= parent.start and argV.end <= parent.end and (parent.end-parent.start) < (shortestSpan.end-shortestSpan.start):
                    shortestSpan=parent
            G.node(self.argID(shortestSpan), shortestSpan.text)
            G.edge(self.argID(argV), self.argID(shortestSpan), color='violet')

        #If visual output requested, then add coref edges determined earlier to a copy of the graph and return that.
        #The returned graph is identical except for nodes created solely as coreference components and the green edges.

        if output:
            H = G.copy()
            for node in corefNodes:
                H.node(node[0],node[1])
            for edge in corefEdges:
                H.edge(edge[0],edge[1],color='green',label=edge[2])
            H.save(filename=(str(hash(self.doc))))

        return G


    def extractSubclaims(self):
        G = nx.nx_pydot.from_pydot(graph_from_dot_data(self.graph.source)[0])
        claimsList = []
        subtrees = []

        # Do networkx things
        # Base of the doc:
        subtreeRoots = list(p[0] for p in G.in_edges(nbunch=self.argID(self.doc[:]))) #Store the subclaim graph roots
        H = nx.subgraph_view(G, filter_node=(lambda n: n != self.argID(self.doc[:]))) #Create a view without the overall text/main root.

        cycling = True
        while cycling:
            try:
                # Create the subclaim graphs - once detaching the whole-text root these are connected components
                subtrees = [H.subgraph(c).copy() for c in nx.weakly_connected_components(H)]
            except nx.HasACycle:
                G.remove_edge(nx.find_cycle(G)[-1])
            else:
                cycling = False
        for sc in subtrees:
            dot=nx.drawing.nx_pydot.to_pydot(subtrees[0])
            s=Source(dot,filename='fish2.gv',format='pdf')
            s.view()
            relRoots = list(filter(lambda x: x in sc, subtreeRoots))
            removedEdges=[]
            for j in relRoots:
                for i in sc.out_edges(j):
                    removedEdges.append((i[0],i[1]))
            sc.remove_edges_from(removedEdges)
            claimsList.append(Claim(self, sc, relRoots))

        return claimsList

class Claim:
    def __init__(self,tlClaim,graph,roots):
        #Claim has Verb and a range of arguments.
        #Also a UVI resolution, and any entities found within.

        self.doc=tlClaim.doc #spacy Doc - just points to parent TLClaim's doc.
        self.graph = graph
        self.roots=roots
        self.kb = KnowledgeBase(self.graph, self.roots,tlClaim.argBaseC)


        return

    def ClaimPrint(self):
        print(self.roots)
        nx.drawing.nx_pydot.write_dot(self.graph, 'fishx')
        return