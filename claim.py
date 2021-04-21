from graphviz import Digraph
import networkx as nx
from pydot import graph_from_dot_data
from logic import KnowledgeBaseHolder, get_span_coref

# Read-in semantically weak adverbs to exclude from conjunctions
with open('adverbStop.txt', 'r') as stopRead:
    stopAdv = []
    for line in stopRead.readlines():
        stopAdv.append(line.replace('\n', ''))

# Gramatical POS tags to drop from comparisons
grammaticalTAG = ["CC", "DT", "EX", "IN", "PDT", "SP", "TO", "UH", "WDT", "WP", "WP$", "WRB"]
grammaticalPOS = ['PUNCT', 'SYM', 'X']


def get_edge_style(label, span):
    if label == "ARGM-TMP" and not span.ents:
        return 'dotted'

    # The below carry either no information or are not present in sufficient numbers to develop rules for them.
    # 1. Semantically weak or infrequent modifier types
    # 2. Semantically weak adverbs (by content)
    # 3. Any modifier whose content is purely grammatical words
    # These are marked on the claim graph with dashed edges.
    if label in ['ARGM-DIS', 'ARGM-LVB', 'ARGM-GOL', 'ARGM-EXT', 'ARGM-ADJ', 'ARGM-REC'] \
            or (label == 'ARGM-ADV' and span.lower_ in stopAdv) \
            or 'R-' in label and all((tok.pos_ in grammaticalPOS or tok.tag_ in grammaticalTAG) for tok in span):
        return 'dotted'
    else:
        return 'solid'


# Converts raw text strings to networkx & nltk-logic safe strings
def arg_id_gen(argV, debug=True):
    if debug:
        # For debug, use an arg generating function that yields closer to the existing string
        k = (str(argV.start) + "X" + str(argV.end) + "X" + (
            argV.text.replace("\"", "").replace(":", "")))  # Sanitize for graph
        j = ''.join(filter(str.isalnum, str(k))).replace(' ', '')  # Pre-emptively sanitize for KB also
        return 'a' + j
    else:  # Otherwise just use a hash
        return 'a' + str(argV.start) + "X" + str(argV.end) + "X" + str(hash(argV.text)).replace("-", "a")


# The main class for derived subclaims. Top-level claim has 1 or more of these.
# Any further information regarding the claim would be furnished here, although this is sparse at the moment.
# Does NOT inherit to prevent confusion between shared sections of graphs.
class Claim:
    def __init__(self, top_claim, graph, roots):
        # Also a UVI resolution, and any entities found within.

        self.doc = top_claim.doc  # spacy Doc - just points to parent TLClaim's doc.

        # Construct constituent KB
        self.kb = KnowledgeBaseHolder(graph, roots, top_claim)

        return


# Class for an individual argument (i.e. a node on the graph or a term in FOL)
# This unifies their linguistic (span + UVI), graphical (ID), and logical (ID)
class ArgNode:

    # Doc is not stored explicitly as it will not be required from this point. Still obtainable from getDoc for debug.
    def __init__(self, ID, doc, span):

        # ID is a unique representation of the argument for use in the knowledge base and within networkX
        self.ID = ID
        self.uvi = doc._.Uvis.get(span, None)
        self.span = span
        self.enabled = False
        return

    # Node is marked as enabled as and only when it is added to the knowledge base - a subset of instantiated argNodes.
    def enable_node(self):
        self.enabled = True
        return

    # Represent with ID for ease of debug
    def __repr__(self):
        return self.ID

    # Provide verb sense in readable format (if one exists)
    def get_uvi_output(self):
        if self.uvi is None:
            return "No Uvi Found"
        else:
            return str(self.uvi)

    def get_doc(self):
        return self.span.doc


# Top level claim - one exists for each input claim (= 1 spacy doc), and consists of multiple subclaims.
class DocClaim:

    def __init__(self, doc_in):
        # argBaseC is a central store of spacy Spans, IDs, VSD, coreferences on a per-argument level.
        self.argBaseC = {}
        self.doc = doc_in

        # Takes a list of OIE subclaims, note that these are not the same as subclaims obtained from the graph.
        # To convert, need to form the abstract meaning representation graph:
        self.graph = self.generate_graph(doc_in._.OIEs)

        # ...then instantiate the subclaims from the graph.
        self.subclaims = self.extract_subclaims()

    # Obtain ID from generating function, and add to central store if not yet there.
    def argID(self, argV, doc=None):
        argID = arg_id_gen(argV)
        if doc is None:
            doc = self.doc
        if argID not in self.argBaseC:
            self.argBaseC[argID] = ArgNode(argID, doc, argV)

        return argID

    # Create graph relating extracted arguments, outputting it if requested. Populates disconnected subgraphs based on
    # sets of verb frames and arguments (OIESubclaims), and then adds shared co-references and implication where a verb
    # is a substring of the argument of another verb.
    def generate_graph(self, oie_subclaims, output=False):

        # The graph is directed as all edges are either implication or property-of relations. Use strict Digraphs to
        # prevent any duplicate edges.
        G = Digraph(strict=True, format='pdf')
        arg_set = set()
        verb_set = set()
        coref_nodes, coref_edges = [], []

        # Create the overall leaf, for which the conjunction of the root verbs will imply:
        G.node(self.argID(self.doc[:]), self.doc.text)

        # Iterate through all extracted relations
        seenRoots=[]
        for claim in oie_subclaims:

            # Plot the verb after having converted it to a node.
            root = claim['V']
            seenRoots.append(root)
            check = ''.join(filter(str.isalnum, str(root))).replace(' ', '')
            if check == "":  # Sometimes some nonsense can be attributed as a verb by oie.
                continue

            # Add arg to graph, with a helpful label for optics.
            G.node(self.argID(root), root.text + '/' + self.argBaseC[self.argID(root)].get_uvi_output())

            for arg_type, argV in claim.items():
                if arg_type != 'V' and argV not in seenRoots:
                    # Create a node for each argument, and a link to its respective verb labelled by its arg type.
                    G.node(self.argID(argV), argV.text + "/" + str(argV.ents) + "/" + str(list(argV.noun_chunks)))
                    G.edge(self.argID(argV), self.argID(root), label=arg_type.replace('-', 'x'),
                           style=get_edge_style(arg_type, argV))
                    # Replace any '-' with 'x' as '-' is a keyword for networkx, but is output by SRL.

                    # Add the argument to the list of arguments eligible to be implied by another subtree.
                    arg_set.add(argV)

                    # Add coreference edges to the graph from the initial text to the entity being coreferenced, only
                    # for the version of the graph that is displayed. Co-references must be omitted from the true
                    # graph as they interfere with splitting into subclaims as the edges becomes bridges. The
                    # coreferences themselves are not lost as they're properties of the doc/spans. They are useful for
                    # illustratory and debugging purposes, and so can be output when requested with output=True.
                    # This is sound to omit from the networkx graph as a coreference does not result in two claims
                    # being co-dependent e.g. 'My son is 11. He likes to eat cake.' - the coreference bridges the two
                    # otherwise separate components when there should be no co-dependence implied. Both are about the
                    # son, but there is not an iff relation between them.
                    arg_corefs = get_span_coref(argV)
                    if output and len(arg_corefs):
                        for cluster in arg_corefs:
                            canonical_reference = cluster.main
                            for inst in cluster.mentions:
                                if inst.start >= argV.start and inst.end <= argV.end and inst != canonical_reference:
                                    coref_nodes.append((self.argID(canonical_reference), canonical_reference.text + "/"
                                                        + str(canonical_reference.ents)))
                                    coref_edges.append((self.argID(argV), self.argID(canonical_reference), inst.text))

                # Add all verbs to the list of eligible roots for subtrees.
                else:
                    verb_set.add(argV)

        # Create 'purple' edges - i.e. edges that link a verb rooting a subtree (of size >= 1) to the argument that they
        # imply. Only one verb can imply any one argument in order to preserve tree-like structure.
        for argV in verb_set:
            shortest_span = self.doc[:]
            for parent in arg_set:
                if argV != parent and argV.start >= parent.start and argV.end <= parent.end and (
                        parent.end - parent.start) < (shortest_span.end - shortest_span.start):
                    shortest_span = parent
            G.node(self.argID(shortest_span), shortest_span.text)
            G.edge(self.argID(argV), self.argID(shortest_span), color='violet')

        # If visual output requested, then add coref edges determined earlier to a copy of the graph and return that.
        # The returned graph is identical except for nodes created solely as coreference components and the green edges.
        if output:
            H = G.copy()
            for node in coref_nodes:
                H.node(node[0], node[1])
            for edge in coref_edges:
                H.edge(edge[0], edge[1], color='green', label=edge[2])
            H.view()

        return G

    # Convert the overall graph of relations into specific subclaims. This is akin to splitting from the overall root
    # into connected subcomponents - each subclaim is independent of the overall claim and is considered separately.
    def extract_subclaims(self):
        # Convert graphviz to networkx.
        G = nx.nx_pydot.from_pydot(graph_from_dot_data(self.graph.source)[0])
        claims_list = []
        subtrees = []

        # Find the subclaim roots - the verbs that the conjunction of implies the documents root.
        subtree_roots = list(p[0] for p in G.in_edges(nbunch=self.argID(self.doc[:])))  # Store the subclaim graph roots
        H = nx.subgraph_view(G, filter_node=(
            lambda n: n != self.argID(self.doc[:])))  # Create a view without the overall text/main root.

        # After removing the root, should have 1+ connected components. If a cycle is detected when trying to find them,
        # remove the last edge that caused it (this rarely happens in practice).
        cycling = True
        while cycling:
            try:
                # Create the subclaim graphs - once detaching the whole-text root these are connected components
                subtrees = [H.subgraph(c).copy() for c in nx.weakly_connected_components(H)]
            except nx.HasACycle:
                G.remove_edge(nx.find_cycle(G)[-1])
            else:
                cycling = False

        # Create a Claim object for each component.
        for subclaim in subtrees:

            # take all roots that form this subclaim (could be multiple if they're connected - this makes the 'tree' not
            # strictly a 'tree').
            rel_roots = list(filter(lambda x: x in subclaim, subtree_roots))
            removed_edges, removed_nodes = [], []

            # Outbound edges from roots are nearly always erroneous, so they are removed as soon as possible:
            # Also remove any cases of nodes that have entirely dashed input.
            for j in rel_roots:
                for i in subclaim.out_edges(j):
                    removed_edges.append((i[0], i[1]))

                for i2 in subclaim.nodes():
                    if len(subclaim.in_edges(i2)) and all(
                            x[2].get('style', '') == 'dotted' for x in subclaim.in_edges(i2, data=True)):
                        removed_nodes.append(i2)
            subclaim.remove_nodes_from(removed_nodes)
            subclaim.remove_edges_from(removed_edges)
            claims_list.append(Claim(self, subclaim, rel_roots))

        return claims_list
