import spacy
from spacy.matcher import Matcher
from spacy.tokens import Doc
import textacy.similarity as ts
from transformer_srl import dataset_readers, models, predictors
import neuralcoref
import Levenshtein

# This is a module rather than a class to enforce singleton behaviour of the models. The nlp_sentencise_only model is used across the
# files and so should be accessible from each module rather than through a single object.

# Higher improves speeds, too high and PyTorch will throw an error.
BATCH_SIZE = 100

# Set as appropriate according to PyTorch config.
CUDA_DEVICE = 0


# Utility to search for key-terms in incoming sentences as to avoid running expensive SRL on irrelevant documents
def doc_usefulness(doc, key_terms):
    for i in doc.ents:
        for j in key_terms[1]:
            # Set a higher threshold if the entity types don't match (to account for the chance of spaCy mislabelling)
            threshold = 0.5 if i.label_ == j.label_ else 0.7
            if ts.token_sort_ratio(i.lower_, j.lower_) > threshold:
                return True

    # Entities are a subset of noun_chunks, so subtract entities from noun_chunks as to not check again.
    leftovers_existing = list(x for x in key_terms[0] if x not in key_terms[1])
    leftovers_incoming = list(x for x in doc.noun_chunks if x not in doc.ents)
    passlist = ['NOUN','PROPN','VERB','X','NUM']
    for i in leftovers_incoming:
        for j in leftovers_existing:
            if (any(tok.pos_ in passlist for tok in i) and any(tok.pos_ in passlist for tok in j)) and (i.similarity(j) > 0.8 or Levenshtein.ratio(i.lower_, j.lower_) > 0.6):
                return True
    return False


# Parse Semantic Role Labelling model response
def srl_parse(srl, doc):
    srl_results = []
    frames = {}
    for e in srl['verbs']:
        srl_spans = tags2spans(e['tags'], doc)
        # Drop extracted relations that don't have a verb and at least 1 numbered argument.
        if len(srl_spans) < 2 or 'V' not in srl_spans or \
                all(x not in srl_spans for x in ['ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4', 'ARG5']) > 0:
            continue
        frames[srl_spans['V']] = e['frame']
        srl_results.append(srl_spans)

    # Store OIEs with the document. Whilst these are only retrieved once, they must be stored within the document
    # to allow spaCy to parallelize.
    doc._.OIEs = srl_results
    doc._.Uvis = frames

    return doc


# Run batch-prediction on srl, useful when spaCy isn't already pipelining SRL (i.e. when processing evidence)
def srl_batch(docs):
    # transformers-SRL doesn't implement batch prediction, so workaround by imitating a json-borne dictionary.
    batched_jsons = []
    srl_sets = []
    index = 0

    # Batch into BATCH_SIZE batches
    while index < len(docs):
        batched_jsons.append(list({'sentence': x.text} for x in docs[index:min(index + BATCH_SIZE, len(docs))]))
        index += BATCH_SIZE

    for i in batched_jsons:
        srl_sets.extend(predictorSRL.predict_batch_json(i))

    return list(map(srl_parse, srl_sets, docs))


# SRL on a single instance, such that spaCy can integrate it with its pipeline.
def srl_pipe(doc):
    oie = predictorSRL.predict_tokenized(list(tok.text for tok in doc))
    doc = srl_parse(oie, doc)
    return doc


#Determine naive quotes presence
def naive_quotes(doc):
    matches = matcher(doc)
    return bool(matches)

#Batch-process documents through the NLP pipeline. Takes raw text documents/claims and returns spaCy docs furnished with
#results of enabled models. For initial claims, entire pipeline runs. For evidence handling, withold SRL model until
#after running doc_usefulness to determine if document will be of any use.
def batch_proc(statements, url_map=None, key_terms=None):
    if url_map is None:
        url_map = {}
    #Hand URL context to spaCy to carry through tagging
    inputData = list(zip(statements, ({'url': url_map.get(s, None)} for s in statements)))

    #If key_terms is None then no search terms have been provided. This will apply for all input claims, but not
    #for input evidence. If there are key_terms, use them to filter the stream of incoming documents. SpaCy does not
    #support filters as part of pipelines, so must be handled outside of pipeline.
    #
    if key_terms is not None:
        with nlp.disable_pipes(['oiePipe']):
            doc_pool = []
            for doc, context in nlp.pipe(inputData, as_tuples=True):
                doc._.url = context['url']
                if doc_usefulness(doc, key_terms) and not naive_quotes(doc) and len(doc) < 500:
                     doc_pool.append(doc)
            docs_out = srl_batch(doc_pool)
    else:
        docs_out = list(x[0] for x in nlp.pipe(inputData, as_tuples=True))
    return docs_out


# Load in spaCy (Tokenizer, Dep parse, NER), and set extensions required for later use.
print("Initiating model load...", end="")
nlp = spacy.load('en_core_web_lg')
print("..")

# Register custom extensions on spaCy documents. These are how information propagates through the pipeline.
Doc.set_extension("Uvis", default={})
Doc.set_extension("OIEs", default={})
Doc.set_extension("url", default='')

# Load in coreference resolver model, register pipe components
coref = neuralcoref.NeuralCoref(nlp.vocab)
nlp.add_pipe(coref, name='neuralcoref')
nlp.add_pipe(srl_pipe, name='oiePipe', last=True)

# Define singleton matcher to detect direct-quotes.
matcher = Matcher(nlp.vocab)
matcher.add("quotes", [[{'ORTH': '"'}, {'IS_ASCII': True, 'OP': '*'}]])

predictorSRL = predictors.SrlTransformersPredictor.from_path("data/srl_bert_base_conll2012.tar.gz", "transformer_srl",
                                                             cuda_device=CUDA_DEVICE, language='en_core_web_lg')
print("Models Loaded")


# Utility function to convert per-token labels from models to spacy-compatible spans.
def tags2spans(tags, docIn):
    spans = {}
    start = 0
    open_span = False
    for index, tag in enumerate(tags):
        if open_span:
            if tag[0] in ('O', 'B') or index == len(tags) - 1:
                end_of_span_adjustment = 1 if index == len(tags) - 1 else 0
                # If hit an O-tagged or B-tagged token and previous arg still open, need to close it and write.
                spans[tags[index - 1][2:]] = docIn[start:index+end_of_span_adjustment]
                # If B-tagged then would set open to False at end of previous arg, before immediately setting to open
                # as this token marks the start of a new arg. Thus, only set open to False if token is O-tagged.
                if tag[0] == 'O':
                    open_span = False

        # Case for going from O-tagged to B-tagged token.
        if tag[0] == 'B':
            # Open new entry and set start index accordingly.
            start = index
            open_span = True

    # Extend any derived spans that would cut off part of an entity (as detected by spaCy's NER model)
    # First, detect if the opening token is (I)nside an entity, and if so extend argument leftward to fully include it.
    for spanK, span in spans.items():
        if docIn[span.start].ent_iob_ == 'I':
            i = span.start - 1
            # Extend leftward until hitting the next 'B' tagged token i.e. the start of the entity.
            while docIn[i].ent_iob_ != 'B':
                i -= 1
            # Update the span with the new start point and existing end point.
            spans[spanK] = docIn[i:span.end]

    # This has to be two loops as the iterable is modified, so need it written back before changing the start index.
    # As above, but for the end of the Span.
    # Note that spaCy defines the 'end' property as 1 + the index of the final token of the span.
    for spanK, span in spans.items():
        if docIn[span.end - 1].ent_iob_ in ['I', 'B'] and span.end + 1 < len(docIn):
            j = span.end
            while docIn[j].ent_iob_ != 'O' and j != span.end:
                j += 1
            spans[spanK] = docIn[span.start:j]
    return spans


if __name__ == 'main':
    print("Run main.py instead.")
