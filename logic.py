from string import punctuation
from collections import deque
from nltk import Prover9, Prover9Command
from spacy.tokens.token import Token
from nltk.sem import Expression as Expr
import tokens
import claim
import evidence

#Setup prover9 interface (singleton)
prover = Prover9()

#NLTK interface doesn't work well with Windows, so specify binary locations directly.
Prover9._prover9_bin = tokens.PROVER9_PATH
Prover9._prooftrans_bin = tokens.PROVER9_TRANSF_PATH


def get_span_coref(span):
    coref_set = set()
    for tok in span:
        if tok._.in_coref:
            for cf in tok._.coref_clusters:
                coref_set.add(cf)
    return list(coref_set)


class KnowledgeBaseHolder:
    # Define core arguments, and those to exclude (Right-args)
    core = ['ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4', 'ARG5']
    other = ['SKIP', 'RxARG0', 'RxARG1']
    c = Expr.fromstring('argF(root)')

    def __init__(self, claim_in, roots, doc_claim):
        self.doc_c = doc_claim
        self.claimG = claim_in
        self.roots = roots
        self.argBaseC = doc_claim.argBaseC  # argBaseC stores association between span and ID
        self.kb = []  # kb is the list of expressions on which inference is run
        self.freeVariableCounter = 0  # A counter to ensure free variables are unique.
        self.searchTerms = []  # List of potential search terms, updated as the knowledge base is populated
        self.argFunc = doc_claim.argID
        self.kb_rules_only = []  # A subset of the knowledge base containing ONLY rules.
        self.kb_rules_only_to_args = {}  # A map between the rules in kb_rules_only and their associated args
        self.kb_leaf_args_to_formula = {}
        self.ruleLength = 0  # Number of rules in the KB - so evidence backtracking knows how far to go.
        self.evidenceMap = {}  # Associates evidence with Span (as they bypass argbaseC)
        self.graph2rules()  # Populate the above on init

    # Get dict of args filtered to include only those which were eventually added to the kb.
    def get_enabled_arg_base_c(self):
        return {k: v for k, v in self.argBaseC.items() if v.enabled}

    # Convert accrued searchTerms into a formed query to send to webCrawler
    def prep_search(self):
        queries = []
        entities = []
        ncs = []

        # If no searchTerms have been accrued then just add the entire claim as one first
        if not self.searchTerms:
            for root in self.roots:
                self.searchTerms.append((root, list(x[0] for x in self.claimG.in_edges(nbunch=root))))

        # For every term, substitute in coreference canonical references. Populate entities + noun_chunk lists for use
        # in post-collection document culling.
        for term in self.searchTerms:
            span_list = sorted(list(self.argBaseC[arg].span for arg in term[1] + [term[0]]), key=lambda x: x.start)
            coref_subd_spans = []
            for span in span_list:
                entities.extend(span.ents)
                ncs.extend(span.noun_chunks)
                used_corefs = []
                new_span = []
                for tok in span:
                    # Don't subtitute any coreferences in which are just pronouns - will never be gaining information.
                    if tok._.in_coref and tok.tag_ in ('PRP', 'PRP$'):
                        coref = tok._.coref_clusters[0]
                        if coref not in used_corefs:
                            new_span.extend(tok for tok in coref.main)
                            used_corefs.append(coref)
                            entities.extend(coref.main.ents)
                        if tok.tag_ == 'PRP$':
                            new_span.append("'s ")
                    else:
                        new_span.append(tok)
                coref_subd_spans.append(''.join((tok.text_with_ws if type(tok) == Token else tok) for tok in new_span))
            queries.append(' '.join(coref_subd_spans).replace('  ', ' ').replace(" 's", "'s"))

        # Remove any queries which are
        if queries:
            q2 = sorted(queries)[0]
        else:
            q2 = None
        return q2, ncs, entities

    # Determines whether edge leads to a 'core' argument (i.e. a named one, and/or one that is not a leaf), or if
    # leads to a modifier (ARGM-) or a leaf argument ('other').
    def mod_or_core(self, edge):
        if len(list(x for x in self.claimG.in_edges(nbunch=edge[0], data=True) if
                    x[2].get('style', '') != 'dotted')) and 'ARGM' not in edge[2].get('label', ''):
            return 'coreInterior'
        if edge[2].get('label', '') in self.core:
            return 'core'
        if edge[2].get('label', 'SKIP') not in self.core + self.other and edge[2].get('style','') != 'dotted':
            if edge[2].get('label', 'SKIP') == 'ARGM-NEG':
                return 'neg'
            else:
                return 'mod'
        else:
            return 'other'

    # Generates a unique free variable for the knowledge base rules to be instantiated with.
    def get_free_var(self):
        modifier = self.freeVariableCounter
        ret_val = 'u' + str(modifier)
        self.freeVariableCounter += 1
        return ret_val

    def add_to_kb(self, text):
        if text[:4] == ' -> ':
            print("No predicate found, KB adding aborted")
            return
        else:
            existing = list(str(x) for x in self.kb)
            exp = Expr.fromstring(text)
            if text not in existing:
                self.kb.append(exp)
            return

    def graph2rules(self):
        # Keep track of seen nodes as to avoid cycles.
        seen = set()
        if len(self.roots) == 0:
            return
        # Create the root implication as the conjunction of the verbs that feed into the root.
        root_impl = self.conj_establish(self.roots, seen) + ' -> argF(root)'
        self.add_to_kb(root_impl)
        self.ruleLength = len(self.kb)

        return

    # For multiple verbs feeding 1 argument, the argument is implied by conjunction of the subtrees rooted at the verbs
    # So to establish an argument which has incoming edges, we must establish all incoming edges.
    def conj_establish(self, roots_in, seen):
        filtered_roots = (y for y in roots_in if len(self.claimG.in_edges(nbunch=y)) > 0)
        inc = list((x, self.establish_rule(x, seen)) for x in filtered_roots)
        inc2 = list(x for x in inc if '()' not in x[1])
        for node, node_rule in inc2:
            interiors = list(x for x in self.claimG.in_edges(nbunch=node, data=True) if x[0] in node_rule)
            filtered_interiors = list(map(self.mod_or_core, interiors))
            if 'coreInterior' in filtered_interiors:
                odls = node_rule.split(" &")[0]
                mini_arglist = []
                mini_leaf_only = []
                root = node_rule.split('(')[0]

                for index, val in enumerate(filtered_interiors):
                    if interiors[index][0] in odls:
                        if val == 'coreInterior':
                            mini_arglist.append(root + "ARG" + str(index))
                        else:
                            mini_arglist.append(interiors[index][0])
                            mini_leaf_only.append(interiors[index][0])
                new_arg = root + "(" + ",".join(mini_arglist) + ")"

                print("NEWARG", new_arg)
                if 'core' not in filtered_interiors:
                    #if up_val is entirely interior args (e.g. asked(asked1,asked2)):
                    # add asked(asked1,asked2) to the kb but nothing else
                        self.add_to_kb(new_arg)

                else:
                    # if up_val has at least 1 interior but not all interior (e.g. asked(asked1, asked2, bolton))
                    # add asked(bolton) to self.kb_rules_only, then if seeing evidence asked(bolton) then add asked(asked1, asked2, bolton) to the kb.
                    miniargleaf = root + "(" + ",".join(mini_leaf_only)+")"
                    print("MINILEAFARG: ", miniargleaf)
                    self.kb_leaf_args_to_formula[miniargleaf] = new_arg
                    self.kb_rules_only.append(miniargleaf)

        rules_only = list(x[1] for x in inc2)
        self.kb_rules_only.extend(rules_only)
        return " & ".join(rules_only)

    # Take a node and establish it as a predicate function, with its arguments being the verb (node)'s arguments.
    def establish_rule(self, root, seen):
        self.argBaseC[root].enable_node()
        seen.add(root)
        negate_predicate = False
        arg_list = []
        modifiers = []
        # Find all edges coming into the root verb, Sort to ensure Arg0 is processed first.
        incoming_edges = sorted(self.claimG.in_edges(nbunch=root, data=True), key=lambda x: x[2].get("label", "Z"))
        for edge in incoming_edges:
            # Find the core args to create the predicate. Modifiers are not permitted in the predicate at this point.
            if edge[2].get('style', '') != 'dotted' and self.mod_or_core(edge) in ['coreInterior', 'core']:
                arg_list.append(edge[0])

        if len(arg_list) > 1:
            self.searchTerms.append((root, arg_list))

        # Now check for any non-leaf entries or modifiers
        count = 0
        for edge in incoming_edges:  # Arg0, Arg1 etc for the root verb.
            if edge[2].get('style', '') == 'dotted':
                continue

            # Is argx an interior node (i.e. has incoming violet/verb-subpart edges:) - NOT modifiers.
            # If so, need to handle the subtree rooted at it (i.e. recurse deeper)

            # Looking at an arg, we're just checking if it has at least 1 violet edge in. Modifiers are not args.
            if len(self.claimG.in_edges(nbunch=edge[0])) > 0 and 'ARGM' not in edge[2].get('label', '') \
                    and edge[0] not in seen:

                # Conjestablish over all incoming violet edges (although usually is just 1)
                up_val = self.conj_establish(list(x[0] for x in (self.claimG.in_edges(nbunch=edge[0]))), seen)
                implied_arg = root+str(len(arg_list))+"ARG"+str(count)+" = "+edge[0]
                if up_val:
                    self.add_to_kb(up_val + ' -> ' + implied_arg)

            elif self.mod_or_core(edge) == 'neg':
                negate_predicate = True

            # Else if it's a modifier
            elif self.mod_or_core(edge) == 'mod':
                # print("mod", edge)
                mod_type = edge[2]['label'].replace("ARGMx", "")
                mod_val_id = edge[0]
                mod_val = self.argBaseC[mod_val_id].span

                if mod_type in ['TMP'] + ['MOD', 'ADV', 'PRP', 'CAU', 'LOC']:
                    if 'never' in mod_val.lower_:
                        negate_predicate = True
                    else:
                        modifiers.append((mod_type, mod_val_id))
                        self.argBaseC[mod_val_id].enable_node()
                elif mod_type in ['DIR', 'PRD', 'MNR']:
                    if not (mod_type == 'MNR' and all(tok.tag_ not in claim.grammaticalTAG for tok in mod_val)):
                        arg_list.append(mod_val_id)

                # print("MODIFIER ", modType+'('+str(argList)+','+str(modValID)+')')

                # If the modifier has incoming edges from verbs that root subtrees (i.e the modifier contains 1+ verbs):
                if len(self.claimG.in_edges(nbunch=edge[0])) > 0 and edge[0] not in seen:
                    up_val = self.conj_establish(list(x[0] for x in (self.claimG.in_edges(nbunch=edge[0]))), seen)

                    # The free value here represents the predicate.
                    implied_arg = mod_type + '(' + self.get_free_var() + ',' + mod_val_id + ')'
                    self.add_to_kb(up_val + ' -> ' + implied_arg)

            # Else its a leaf, so just continue without adding any extra rules or modifier

            # Increase the count as we move to the count-th argument (i.e. exclude Modifiers -
            # theoretically this shouldn't matter as count falls out of use after numbered arguments, which come first)
            if 'ARGM' not in edge[2].get('label', ''):
                count += 1

        # Form the predicate - have to do this now so we can add modifiers on the next pass of the edges.
        predicate = root + str(len(arg_list)) + '(' + ','.join(arg_list) + ')'

        for arg in arg_list:
            self.argBaseC[arg].enable_node()

        if negate_predicate:
            predicate = '-' + predicate

        # Add the predicates
        old_pred = predicate
        self.kb_rules_only_to_args[old_pred] = []
        for m in modifiers:
            modifier_text = m[0] + old_pred.translate(str.maketrans('', '', punctuation)) + '(' + m[1] + ')'
            self.kb_rules_only_to_args[old_pred].append(modifier_text)
            predicate += " & " + modifier_text

        return predicate

    # Run inference with current KB population.
    def prove(self):
        if len(self.kb) == self.ruleLength:
            return False, []
        ev_docs = []
        for index, x in enumerate(self.kb):
            print(index, ":", x)
        print("Attempting proof...")
        prover9_prover = Prover9Command(goal=Expr.fromstring('argF(root)'), assumptions=self.kb, prover=prover)
        p1 = prover9_prover.prove(verbose=True)
        if p1:
            prf= prover9_prover.proof()
            print(prf)
            if 'argF(root)' not in prf:
                return False, []
            # Parse unhelpful text output format
            prf_lines = prf.split("% Given clauses")[1]
            prf_lines_2 = list(x for x in prf_lines.split(".") if '[' in x)
            prf_lines_3 = list(x for x in prf_lines.split("\n") if '[' in x)
            evtrip = {}
            final = 0
            for index, x in enumerate(prf_lines_3):
                next_hop = prf_lines_2[index]
                if '(' in next_hop:
                    next_hop = next_hop.split('(')[1].split(')')[0]
                    if ',' in next_hop:
                        next_hop = list(int(x) for x in next_hop.split(',') if x.isnumeric())
                    else:
                        next_hop = [int(next_hop)]
                else:
                    next_hop = []

                split2 = x.split(".")[0].split(" ")
                evtrip[int(split2[0])] = (" ".join(split2[1:]),next_hop)
                if index == len(prf_lines_3)-1:
                    final = int(split2[0])

            # Obtain proof from backtracker
            path = backtracker(evtrip, final)


            # Write proof derivation
            for index, i in enumerate(path):
                if 0 < index < len(path) - 1:
                    ev_docs.append("-->--")
                if type(i) is not tuple:
                    if i in self.evidenceMap:
                        best = max(enumerate(self.evidenceMap[i]), key=lambda x: evidence.lcs(x[1][1],self.doc_c.doc))[1][1]
                        ev_docs.append(best.text + " @ " + best._.url)
            return p1, ev_docs
        else:
            return p1, []


# Backtrack through derivation to extract the proof, using ruleLength to determine when 'leaves' are reached.
def backtracker(prf, start):
    path = []
    queue = deque([start])
    while len(queue):
        k = queue.popleft()
        step = prf[k][1]
        if step:
            queue.extend(step)
        else:
            if prf[k][0] not in path and not any(x in prf[k][0] for x in ['argF(root)',"=","->"]):
                split_clauses = list(x for x in prf[k][0].split(" ") if len(x) > 2)
                path.extend(split_clauses)
    path.reverse()
    return path
