import time

from graphviz import Digraph, Source
import networkx as nx
from pydot import graph_from_dot_data
from logic import KnowledgeBase, getSpanCoref

with open('adverbStop.txt','r') as stopRead:
    stopAdv = []
    for l in stopRead.readlines():
        stopAdv.append(l.replace('\n',''))

grammaticalTAG = ["CC", "DT", "EX", "IN", "PDT", "SP", "TO", "UH", "WDT", "WP", "WP$", "WRB"]
grammaticalPOS = ['PUNCT', 'SYM', 'X']

def getEdgeStyle(label, span):
    if label in ['RARG0','AND', 'OR', 'IF', 'ARGM-DIS'] \
            or (label == 'ARGM-ADV' and span.lower_ in stopAdv)\
            or 'R-' in label and all((tok.pos_ in grammaticalPOS or tok.tag_ in grammaticalTAG) for tok in span):
        return 'dotted'
    else:
        return 'solid'

#Converts raw text strings to networkx & nltk-logic safe strings
def argIDGen(argV):
    debug = True
    if debug:
        #For debug, use an arg generating function that yields closer to the existing string
        k=(str(argV.start) + "X" + str(argV.end) +"X" +(argV.text.replace("\"", "").replace(":",""))) #Sanitize for graph
        j = ''.join(filter(str.isalnum, str(k))).replace(' ', '') #Pre-emptively sanitize for KB also
        return j
    else: #Otherwise just use a hash
        return (str(argV.start) + "X" + str(argV.end) +"X" +str(hash(argV.text)).replace("-","a"))

class Claim:
    def __init__(self,tlClaim,graph,roots):
        #Claim has Verb and a range of arguments.
        #Also a UVI resolution, and any entities found within.

        self.doc=tlClaim.doc #spacy Doc - just points to parent TLClaim's doc.
        self.graph = graph
        self.roots=roots
        self.kb = KnowledgeBase(self.graph, self.roots,tlClaim.argBaseC)


        return

    #todo
    def ClaimPrint(self):
        return ""

class argNode():

    #Doc is not stored explicitly as it will not be required from this point. Still obtainable from getDoc for debug.
    def __init__(self,ID,doc,span):
        self.ID = ID
        self.uvi = doc._.Uvis.get(span, None)
        self.span = span
        return

    def __repr__(self):
        return self.getUviOutput() + " " + self.span.text

    #Provide verb sense in readable format (if one exists)
    def getUviOutput(self):
        if self.uvi is None: return "No Uvi Found"
        else: return str(self.uvi)

    def getDoc(self):
        return self.span.doc

#Top level claim - one exists for each input claim (= 1 spacy doc), and consists of multiple subclaims.
class docClaim:

    def __init__(self, docIn):
        #argBaseC is a central store of spacy Spans, IDs, VSD, coreferences on a per-argument level.
        self.argBaseC = {}
        self.doc = docIn

        # Takes a list of OIE subclaims, note that these are not the same as subclaims obtained from the graph.
        #To convert, need to form the abstract meaning representation graph:
        self.graph = self.generateCG(docIn._.OIEs)

        #...then extract the subclaims from the graph.
        self.subclaims = self.extractSubclaims()

    #todo
    def printTL(self):
        return ""

    #Obtain ID from generating function, and add to central store if not yet there.
    def argID(self,argV):
        argID = argIDGen(argV)

        if argID not in self.argBaseC:
            self.argBaseC[argID] = argNode(argID, self.doc, argV)

        return argID

    #Create graph relating extracted arguments, outputting it if requested.
    def generateCG(self,OIEsubclaims, output=False):
        #The graph is directed as all edges are either implication or property-of relations. Use strict Digraphs to
        #prevent any duplicate edges.
        G = Digraph(strict=True,format='pdf')
        argSet= set()
        verbSet = set()
        corefNodes, corefEdges = [], []

        #Create the overall leaf, for which the conjunction of the root verbs will imply:
        G.node(self.argID(self.doc[:]), self.doc.text)

        #Iterate through all extracted relations
        for claim in OIEsubclaims:

            #Plot the verb after having converted it to a node.
            root=claim['V']
            check = ''.join(filter(str.isalnum, str(root))).replace(' ', '')
            if check =="": #Sometimes some nonsense can be attributed as a verb by oie.
                continue
            G.node(self.argID(root), root.text + '/' + self.argBaseC[self.argID(root)].getUviOutput())

            for argK, argV in claim.items():
                if argK != 'V':
                    #Create a node for each argument, and a link to its respective verb labelled by its arg type.
                    G.node(self.argID(argV), argV.text + "/" + str(argV.ents))
                    G.edge(self.argID(argV), self.argID(root), label=argK.replace('-', 'x'), style=getEdgeStyle(argK,argV))
                    # Replace any '-' with 'x' as '-' is a keyword for networkx, but is output by allennlp. This label
                    #has no usage past displaying this output.

                    #Add the argument to the list of arguments elligible of being implied by another subtree.
                    argSet.add(argV)

                    #Add coreference edges to the graph from the initial text to the entity being coreferenced, but only
                    #for the version of the graph that is displayed. Co-references must be omitted from the true
                    #graph as they interfere with splitting into subclaims as the edges becomes bridges. The
                    #coreferences themselves are not lost as they're properties of the doc/spans. They are useful for
                    #illustratory and debugging purposes, and so can be output when requested with output=True.
                    #This is deemed sound to omit from the networkx graph as a coreference does not result in two claims
                    #being co-dependent e.g. 'My son is 11. He likes to eat cake.' - the coreference bridges the two
                    #otherwise separate components when there should be no co-dependence implied. Both are about the
                    #son, but there is not an iff relation between them.
                    #print(argV, "  cf ", argV._.coref_cluster)
                    argVcf = getSpanCoref(argV)
                    if output and len(argVcf):
                        for cluster in argVcf:
                            mainSpan = cluster.main
                            for inst in cluster.mentions:
                                #print()
                                #print(argV, "[",argV.start,":",argV.end,"] , ",inst, " [",inst.start,":",inst.end,"]")
                                if inst.start >= argV.start and inst.end <= argV.end and inst != mainSpan:
                                    #todo this should point back around to another arg where it encapsulates it.
                                    corefNodes.append((self.argID(mainSpan), mainSpan.text + "/" + str(mainSpan.ents)))
                                    #print(argV, "/", inst, " -> ", mainSpan)
                                    corefEdges.append((self.argID(argV), self.argID(mainSpan), inst.text))

                #Add all verbs to the list of eligible roots for subtrees.
                else:
                    verbSet.add(argV)

        #Add connectives
        for edge in self.doc._.ConnectiveEdges:
            G.node(self.argID(edge.start), edge.start.text)
            G.node(self.argID(edge.end), edge.end.text)
            G.edge(self.argID(edge.start), self.argID(edge.end), color=edge.colours[edge.connType], label=edge.connType, style=getEdgeStyle(edge.connType, edge.end))

            if edge.connType == 'IF':
                raise NotImplementedError
                #todo fix these 3 lines - the 'TV' including example ends up with having not a verb as the parent of the root.
                #argSet.add(edge.start)
                #argSet.add(edge.end)
                #verbSet.add(edge.end)
                None

            elif edge.connType == 'OR':
                verbSet.add(edge.start)
                verbSet.add(edge.end)

        #Handle 'purple' edges - i.e. edges that link a verb rooting a subtree (of size >= 1) to the argument that they
        #imply. Only the shortest possible existing argument can be implied by a root, otherwise the tree becomes less tree-like.
        #(Strictly speaking the graph is not always a tree, but adidng edges other than to the shortest encapsulating
        #argument increases complexity and adds no information).
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
            s=Source(H,filename='fish2.gv',format='pdf')
            s.view()
            #H.save(filename=(str(hash(self.doc))))

        return G


    #Convert the overall graph of relations into specific subclaims. This is akin to splitting from the overall root
    #into connected subcomponents - each subclaim is independent of the overall claim and should be considered
    #separately. The overall veracity of the input claim is then determined in light of the makeup of the subclaims' veracity.
    def extractSubclaims(self):
        #Convert graphviz to networkx.
        G = nx.nx_pydot.from_pydot(graph_from_dot_data(self.graph.source)[0])
        #print(G.edges(data=True))

        """dot = nx.drawing.nx_pydot.to_pydot(G)
        s = Source(dot, filename='fisht.gv', format='pdf')
        time.sleep(0.5)
        s.view()"""

        claimsList = []
        subtrees = []

        # Find the subclaim roots - the verbs that the conjunction of implies the documents root.
        subtreeRoots = list(p[0] for p in G.in_edges(nbunch=self.argID(self.doc[:]))) #Store the subclaim graph roots
        H = nx.subgraph_view(G, filter_node=(lambda n: n != self.argID(self.doc[:]))) #Create a view without the overall text/main root.

        #After removing the root, should have 1+ connected components. If a cycle is detected whilst trying to find them,
        #remove the last edge that caused it (this rarely happens in practice).
        cycling = True
        while cycling:
            try:
                # Create the subclaim graphs - once detaching the whole-text root these are connected components
                subtrees = [H.subgraph(c).copy() for c in nx.weakly_connected_components(H)]
            except nx.HasACycle:
                G.remove_edge(nx.find_cycle(G)[-1])
            else:
                cycling = False

        #Create a Claim object for each component.
        for sc in subtrees:
            dot=nx.drawing.nx_pydot.to_pydot(subtrees[0])
            s=Source(dot,filename='fish2.gv',format='pdf')
            #s.view()
            #take all roots that form this subclaim (could be multiple if they're connected - this makes the 'tree' not
            #strictly a 'tree').
            relRoots = list(filter(lambda x: x in sc, subtreeRoots))
            removedEdges=[]
            removedNodes=[]
            #Outbound edges from roots are nearly always erroneous, so they are removed as soon as possible:
            #Also remove any cases of nodes that have entirely dashed input.
            for j in relRoots:
                for i in sc.out_edges(j):
                    removedEdges.append((i[0],i[1]))

                for i2 in sc.nodes():
                    if len(sc.in_edges(i2)) and all(x[2].get('style','') == 'dotted' for x in sc.in_edges(i2,data=True)):
                        removedNodes.append(i2)
            sc.remove_nodes_from(removedNodes)
            sc.remove_edges_from(removedEdges)
            claimsList.append(Claim(self, sc, relRoots))

        return claimsList