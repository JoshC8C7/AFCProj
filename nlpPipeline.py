import spacy
from spacy.tokens import Doc, Span
from claim import getCorefs
import connectives as con
import pathlib
import os
from pprint import pprint
from transformer_srl import dataset_readers,models,predictors
import  neuralcoref
#This is a module rather than a class to enforce singleton behaviour of the models. The nlp model is used across the
#files and so should be accessible from each module rather than through a single object.


def oiePipe(doc):
    oie = predictorOIE.predict(doc.text)
    # Parse Open Information Extraction model response & combine with Verb Sense information.
    oieSubclaims = []
    frames={}
    for e in oie['verbs']:
        oiespans = tags2spans(e['tags'], doc)
        # Drop extracted relations that don't have a verb and at least 1 numbered argument.
        if len(oiespans) < 2 or 'V' not in oiespans or \
                all(x not in oiespans for x in ['ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4', 'ARG5']) > 0:
            print("axing ", oiespans)
            continue
        #print(oiespans['V'], " ",e['frame'], " ", e['frame_score'])
        frames[oiespans['V']] = e['frame']
        oieSubclaims.append(oiespans)

    #Store OIEs with the document. Whilst these are only retrieved once, they must be stored within the document
    #to allow spaCy to parallelize.
    doc._.OIEs = oieSubclaims
    doc._.Uvis = frames

    return doc

def corefPipe(doc):
    # Parse coreference resolution model response into corefSpans = {text (Span): what its coreferencing (Span)}
    print("new ", doc._.coref_clusters)
    #todo corefSpans takes format of {their: Nancy Pelosi} i.e. k=coreference v=thing its pointing to

    """if len(coref['clusters']):
        for index, value in enumerate(coref.get('predicted_antecedents',[])):
            if value != -1:
                source = coref['top_spans'][index]
                dest = coref['top_spans'][value]
                corefSpans[doc[source[0]:source[1] + 1]] = doc[dest[0]:dest[1] + 1]
    """
    doc._.DCorefs = corefSpans
    print("OLD ", corefSpans)

    return doc



# Load in spaCy (Tokenizer, Dep parse, NER), and set extensions required for later use.
print("Initiating model load...",end="")
nlp = spacy.load('en_core_web_lg')
print("..")
Doc.set_extension("DCorefs", default={})
Doc.set_extension("Uvis", default={})
Doc.set_extension("OIEs", default={})
Doc.set_extension("ConnectiveEdges", default=[])
Span.set_extension("SCorefs", getter=getCorefs)
# Run connective extractor over input text and store result in doc._.extract_connectives.
nlp.add_pipe(oiePipe,name='oie',last=True)
neuralcoref.add_to_pipe(nlp)
nlp.add_pipe(corefPipe,name='coref',last=True)
nlp.add_pipe(con.extractConnectives, name='extract_connectives', last=True)

predictorOIE = predictors.SrlTransformersPredictor.from_path("data/srl_bert_base_conll2012.tar.gz", "transformer_srl")
print("Models Loaded")

def batchProc(statementSet):
    docs = list(nlp.pipe(list(statementSet)))
    return docs

def tags2spans(tags,docIn):
    spans={}
    start=0
    open=False
    for index,tag in enumerate(tags):
        if open:
            if tag[0] in ('O','B') or index==len(tags)-1:
                #If hit an O-tagged or B-tagged token and previous arg still open, need to close it and write.
                spans[tags[index - 1][2:]] = docIn[start:index]
                #If B-tagged then would set open to False at end of previous arg, before immediately setting to open
                #as this token marks the start of a new arg. Thus, only set open to False if token is O-tagged.
                if tag[0] == 'O':
                    open=False

        #Case for going from O-tagged to B-tagged token.
        if tag[0] == 'B':
            # Open new entry and set start index accordingly.
            start = index
            open = True

    #Extend any derived spans that would cut off part of an entity (as detected by spaCy's NER model)
    #First, detect if the opening token is (I)nside an entity, and if so extend argument leftward to fully include it.
    for spanK, span in spans.items():
        if docIn[span.start].ent_iob_ == 'I':
            i=span.start-1
            #Extend leftward until hitting the next 'B' tagged token i.e. the start of the entity.
            while docIn[i].ent_iob_ != 'B':
                i-=1
            #Update the span with the new start point and existing end point.
            spans[spanK] = docIn[i:span.end]

    #This has to be two loops as the iterable is modified, so need it written back before changing the start index.
    #As above, but for the end of the Span.
    #Note that spaCy defines the 'end' property as 1 + the index of the final token token of the span.
    for spanK, span in spans.items():
        if docIn[span.end-1].ent_iob_ in ['I','B'] and span.end+1 < len(docIn):
            j = span.end
            while docIn[j].ent_iob_ != 'O' and j!= span.end:
                j += 1
            spans[spanK] = docIn[span.start:j]

    return spans

if __name__ == 'main':
    print("Running this file has no effect! Run main.py instead.")