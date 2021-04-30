from string import punctuation, printable

import pylcs
from nltk.corpus import wordnet as wn
from nltk.corpus.reader import WordNetError
from textacy import similarity
from word2number.w2n import word_to_num as w2n
import re
import claim
import numpy as np
import textdistance.algorithms as tda
import json
import nlpPipeline
from spacy.lang.en import English, STOP_WORDS
import requests

# Setup wikidata API
wd_endpoint = "https://www.wikidata.org/w/api.php?action=wbsearchentities&search={0}&language=en&format=json&limit=3"

# Import spacy sentencizer only for pre-processing/lightweight culling of irrelevant documents.
nlp_sentencise_only = English()
nlp_sentencise_only.add_pipe(nlp_sentencise_only.create_pipe("sentencizer"))

# Read in PropBank to WordNet dictionary
with open('data/pb2wn.json', 'r') as inFile:
    propbank_to_wordnet = json.load(inFile)

def lcs(a,b):
    a_stripped = a.text
    b_stripped = b.text
    return pylcs.lcs2(a_stripped,b_stripped)/(max(len(a_stripped),len(b_stripped)))


# Compare two entities via wikidata query. Checks for any intersection between two sets of 3 entities (UIDs) returned
# by a wikidata query, and caches them due to the high chance of a query being repeated.
def wikidata_compare(node1, node2, cache):
    # Only compare entities where at least one token is labelled as a (proper) noun.
    if all(x.pos_ not in ('PROPN', 'NOUN') for x in node1) or all(
            y.pos_ not in ('PROPN', 'NOUN') for y in node2) or (
            node1.label_ and node2.label_ and node1.label_ != node2.label_):
        return False

    # If searching on noun chunks instead, just look at their root (which will be the noun).
    if node1.label_ is None:
        node1 = node1.root
        node2 = node2.root

    # Fetch from wikidata
    id1 = wiki_fetch(node1, cache)
    id2 = wiki_fetch(node2, cache)

    # Return if any common entity
    return any(x in id2 for x in id1)


# Interface with cache and/or fetch from wikidata
def wiki_fetch(node, cache):
    if node.text in cache and cache[node.text] is not None:
        id_list = cache[node.text]
    else:
        wiki_data = requests.get(wd_endpoint.format(node.text.replace(" ", "%20")))
        try:
            wiki_data_parsed = wiki_data.json().get('search', [])
        except json.decoder.JSONDecodeError:
            return []
        else:
            id_list = list(x.get('id', '') for x in wiki_data_parsed)
            cache[node.text] = id_list
    return id_list


# Compare two entities which are numbers
def num_compare(node1, node2):
    if node1.label_ == node2.label_ == 'CARDINAL':
        numerical_node1 = re.findall(r'[\d.]+', node1.text)
        numerical_node2 = re.findall(r'[\d.]+', node2.text)

        # Attempt text to digit conversion (one million -> 1,000,000)
        try:
            node1_val = float(w2n(node1.text)) * (1.0 if not numerical_node1 else float(numerical_node1[0]))
        except ValueError:
            if numerical_node1:
                node1_val = float(numerical_node1[0])
            else:
                return True
        try:
            node2_val = float(w2n(node2.text)) * (1.0 if not numerical_node2 else float(numerical_node2[0]))
        except ValueError:
            if numerical_node2:
                node2_val = float(numerical_node2[0])
            else:
                return True

        # Accept comparison if within 30% either way.
        return node2_val * 0.7 < node1_val < node2_val * 1.3
    else:
        return True


# Process incoming evidence into the knowledge base, if relevant. Attempts to co-resolve with entities
# already in the knowledge base, where possible.
def process_evidence(subclaim, ncs, entities, sources):
    # Cache queries sent to wikidata as is good practice.
    wiki_cache = {}

    # Strip existing to compare with incoming - i.e. don't accept a direct copy of the claim being checked, as evidence.
    stripped_existing = subclaim.doc.text.translate(str.maketrans('', '', punctuation)).lower()

    # Pre-process evidence
    evidence = receive_doc(sources, subclaim.doc)
    url_map = dict((x[1], x[0]) for x in evidence)

    # Batch run evidence through NLP pipeline
    ev_docs = nlpPipeline.batch_proc(list(x[1] for x in evidence), url_map, (ncs, entities))

    sought_v = set()
    verb_matched_relations = []

    # Obtain list of verbs to co-resolve with
    for x in subclaim.kb.argBaseC.values():
        if x.uvi is not None:
            sought_v.add(x)

    for doc in ev_docs:
        if stripped_existing == doc.text.translate(str.maketrans('', '', punctuation)).lower():
            continue

        # Extract verb frames to be passed to verb sense comparison function
        srl_accum = {}
        uvi_matched_srls = {}
        for srl in doc._.OIEs:
            srl_accum[srl['V']] = srl

        # Co-resolve two possible verbs, if possible.
        if srl_accum:
            path_matches, sentiment_matches = verb_match(sought_v, srl_accum, doc._.Uvis)
            if path_matches:
                for p in path_matches:
                    # Obtain rich arg node rather than just the ID, then form dictionary with key as the verb node, and
                    # values as a list of incoming relations whose verb has co-resolved with the key verb node.
                    rich_arg_node = subclaim.kb.get_enabled_arg_base_c().get(p[0], None)
                    if rich_arg_node is not None:
                        if rich_arg_node in uvi_matched_srls:
                            uvi_matched_srls[rich_arg_node].append(p[1])
                        else:
                            uvi_matched_srls[rich_arg_node] = [p[1]]
            # If any verb matches are found, attempt to resolve nouns.
            noun_match(uvi_matched_srls, subclaim, doc, wiki_cache, sentiment_matches)

    return verb_matched_relations


# Obtain coreferences for the input span
def coref_collect(span):
    corefs = []
    for tok in span:
        corefs.extend((x.main for x in tok._.coref_clusters))
    return corefs


# Compare two nouns on a mix of metrics.
def node_compare(inc_node_label, inc_node_span, exst_rich_arg, wiki_cache):
    # Baseline embedding similarity required, and cannot include nodes with an argument type that's dotted (excluded)
    if inc_node_label != 'V' and claim.get_edge_style(inc_node_label, inc_node_span) != 'dotted':
        icorefs, ecorefs = coref_collect(inc_node_span), coref_collect(exst_rich_arg.span)
        # If No entities in either noun, then compare if embeddings are close (by cosine similarity).
        if not (inc_node_span.ents or exst_rich_arg.span.ents) and compute_similarity(inc_node_span, exst_rich_arg.span) > 0.8:
            print(inc_node_span, "/ ", exst_rich_arg.span, "  l1")

            return True
        else:
            # Check if more noun-chunks are alike than are dissimilar, based on levenshtein.
            similar_count, dissimilar_count = 0, 0

            # Existing MUST come first here, as an incoming can provide more information than the existing, but not less.
            # E.g. existing: the democrat lead campaign of domestic terrorism, incoming: democrats.
            # If iterating through incoming first, democrats will match and the iteration completed with 100% matches.
            # If iterating through existing first, democrats will match but noun of the other nouns will -> ~30%.
            for x in exst_rich_arg.span.noun_chunks:
                for y in inc_node_span.noun_chunks:
                    if lcs(x.root,y.root) > 0.35 and compute_similarity(x,y) > 0.57:
                        similar_count += 1
                        break
                else:
                    dissimilar_count += 1

            if similar_count > dissimilar_count:
                return True
            #NCs--------------------------/\-----------------------------/\
            #ENTS-------------------------\/-----------------------------\/----------------------------------\/


            # If entities are present in both, require sane length and for numbers to be sufficiently similar (or the
            # comparison return True as the entities passed are not elligible), and one of levenshtein or wikidata
            # coresolution to exhibit sufficient similarity.
            cb1 = 0
            cb2 = 0
            z = (
                num_compare(x, y) and lcs(x,y) > 0.39 or wikidata_compare(x, y, wiki_cache)
                for x in
                inc_node_span.ents + icorefs for y in set(exst_rich_arg.span.ents + ecorefs))
            for x in z:
                if x:
                    cb1 += 1
                else:
                    cb2 += 1

            # print(inc_node_span, "/", exst_rich_arg.span, c1, "/", c2)
            if cb1 > cb2:
                return True

    # If not matched on any criteria, then return False.
    return False


def noun_match(path_matches, subclaim, incoming_doc, wiki_cache, sentiment_matches):
    for kb_function in subclaim.kb.kb_rules_only:  # Iterate over KB rules
        kb_function = kb_function.replace('-', '')
        kb_function_split = kb_function.split('(')
        kb_predicate_id, function_arity, kb_function_args = kb_function_split[0][:-1], kb_function_split[0][-1:], \
            kb_function_split[1].split(')')[0].split(',')
        kb_predicate = subclaim.kb.get_enabled_arg_base_c().get(kb_predicate_id, None)
        combined_matches = path_matches.get(kb_predicate, []) + sentiment_matches.get(kb_predicate_id, [])
        for match in combined_matches:  # Iterate over incoming oies
            negate_predicate = False
            matched_nouns = [False] * int(function_arity)
            sorted_match = list(sorted(match.items()))
            for index, existing_arg in enumerate(kb_function_args):
                existing_span = subclaim.kb.get_enabled_arg_base_c().get(existing_arg, None)

                if existing_span is None:
                    continue
                for inc_arg_label, inc_arg_value in sorted_match:
                    if node_compare(inc_arg_label, inc_arg_value, existing_span, wiki_cache):
                        # print(inc_node_span, "-> ", exst_rich_arg.span)
                        matched_nouns[index] = True
                    # else:
                    # print(inc_node_span, "-X-> ", exst_rich_arg.span)

            if sum(1 for i in matched_nouns if i) / int(function_arity) >= 0.66:
                modifiers = []
                if 'ARGM-NEG' in match:
                    negate_predicate = True
                for mod in subclaim.kb.kb_rules_only_to_args.get(kb_function.split(" ")[0],[]):
                    # for this predicate, are there modifiers listed? iterate over them
                    # note that some won't have assocaited modifier lists because they were internal nodes and so
                    # were added during establishrule, rather than via conjestablish.
                    modifier_node = subclaim.kb.get_enabled_arg_base_c().get(mod.split('(')[1].replace(')', ''),
                                                                             None)  # Are these modifiers enabled?
                    if modifier_node is None:
                        continue
                    for inc_arg_label, inc_arg_value in sorted_match:

                        if node_compare(inc_arg_label, inc_arg_value, modifier_node, wiki_cache):
                            modifiers.append(mod)

                established_arguments = []
                for index, ent in enumerate(matched_nouns):
                    established_arguments.append(kb_function_args[index])

                new_predicate = kb_predicate_id + function_arity + '(' + ','.join(established_arguments) + ')'
                if negate_predicate:
                    new_predicate = '-' + new_predicate

                subclaim.kb.evidenceMap[new_predicate] = incoming_doc
                for mod in modifiers:
                    new_predicate += ' &' + mod
                    subclaim.kb.evidenceMap[mod] = incoming_doc
                subclaim.kb.add_to_kb(new_predicate)
                print("NEW EVIDENCE: ", new_predicate, "----->", incoming_doc, " @ ", incoming_doc._.url)

    return


# Compute similarity between two spans by cosine distance, ignoring stop words.
def compute_similarity(span1, span2):
    span1_vecs = np.array([tok.vector for tok in span1 if tok.text not in STOP_WORDS])
    span2_vecs = np.array([tok.vector for tok in span2 if tok.text not in STOP_WORDS])
    if not span1_vecs.size or not span2_vecs.size:
        return 0.0
    else:
        span1_vecs_mean = span1_vecs.mean(axis=0)
        span2_vecs_mean = span2_vecs.mean(axis=0)
        denom = np.linalg.norm(span1_vecs_mean) * np.linalg.norm(span2_vecs_mean)
        if denom == 0.0:
            return False
        else:
            return np.dot(span1_vecs_mean, span2_vecs_mean) / denom


# todo
# Takes ALL existing nodes (before looking at all their UVis), and ALL incoming oies (ditto), and look for  uvi matches.
# Returns pairs of (ID of existing node that matches uvi, incoming OIE which matched it)
def verb_match(existing, incoming, uvi_map):
    from nltk.corpus import sentiwordnet as sn
    path_matches = []
    senti_matches = {}
    for jk, j in incoming.items():
        jm = propbank_to_wordnet.get(uvi_map[jk], None)
        if jm is not None:
            for jmm in jm:
                try:
                    wnj = wn.synset(jmm)
                except WordNetError:
                    continue
                for i in existing:
                    im = propbank_to_wordnet.get(i.uvi, None)
                    if im is not None:
                        for imm in im:
                            try:
                                wni = wn.synset(imm)
                            except WordNetError:
                                continue
                            sim = (wn.path_similarity(wni, wnj))
                            if sim > 0.5:
                                if (i.ID, j) not in path_matches:
                                    path_matches.append((i.ID, j))

                            else:
                                isent = sn.senti_synset(imm)
                                jsent = sn.senti_synset(jmm)
                                if ((any(x in imm for x in ['do.v', 'be.v', 'have.v']) or (
                                        isent.pos_score() > 0.3 and not isent.neg_score()))
                                        and (any(x in jmm for x in ['do.v', 'be.v', 'have.v']) or (
                                                jsent.pos_score() > 0.3 and not jsent.neg_score()))):
                                    if i.ID in senti_matches:
                                        if j not in senti_matches[i.ID]:
                                            senti_matches[i.ID].append(j)
                                    else:
                                        senti_matches[i.ID] = [j]
    return list(path_matches), senti_matches


# Preprocess input docs
def receive_doc(sources, doc_text):
    sentences = []
    for source in sources:
        url = source[0]
        # Filter out non-printable characters.
        cleansed_sent = ''.join(filter(lambda x: x in printable and x not in ['{', '}'], source[1]))
        if len(cleansed_sent) < 5:
            continue
        # Sentencize to split doc up and allow for granular filtering, remove extra long sentences.
        doc = nlp_sentencise_only(cleansed_sent)
        for sent in doc.sents:
            if len(sent) < 150 and sent.text.lower() not in doc_text.text.lower():
                sentences.append((url, sent.text.replace("\n", "")))

    return sentences
