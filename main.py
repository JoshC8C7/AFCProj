from allennlp.predictors.predictor import Predictor
import pandas as pd
import os
import spacy
from spacy.tokens import Doc, Span
from flair.data import Sentence
from flair.models import SequenceTagger
from claim import TLClaim, getCorefs, argIDGen
import connectives as con
import pathlib


def main():
    pathlib.PosixPath = pathlib.WindowsPath #Small hack to handle AllenNLP & Flair not being compatible with windows

    #Load in spaCy (Tokenizer, Dep parse, NER), and set extensions required for later use.
    nlp=spacy.load('en_core_web_lg')
    Doc.set_extension("DCorefs",default={})
    Doc.set_extension("Uvis", default={})
    Doc.set_extension("ConnectiveEdges", default=[])
    Span.set_extension("SCorefs",getter=getCorefs)
    #Run connective extractor over input text and store result in doc._.extract_connectives.
    nlp.add_pipe(con.extractConnectives,name='extract_connectives',last=True)

    #Read in input claims
    df=pd.read_table(os.path.join("data","Politihop","Politihop_train.tsv"),sep='\t')

    #The input claims data has multiple repetitions of each text due to containing multiple verifiable claims. This
    #is handled later so for now the text must be de-duplicated. Other text pre-processing/cleansing occurs here.
    statementSet = set()
    for s in df['statement']:
        while not s[0].isalpha():
            s=s[1:]
        if s.split(" ")[0].lower() == "says":
            #todo instead, stick the person speaking on the front here
            s=(s.partition(" "))[1]
        #Allows for filtering to debug specific example.
        if (False or 'bastard' in s) and any(x !=" " for x in s):
            statementSet.add(s)

    #Load in AllenNLP (OIE, Coref), Flair (WSD) models.
    predictorOIE = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz")
    predictorCoref = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz")
    tagger = SequenceTagger.load('frame')
    print("Models Loaded")

    for s in statementSet:
        print(s)
        #Run respective models over statement
        doc=nlp(s)
        oie = predictorOIE.predict(s)
        coref = predictorCoref.predict(s)
        sent1=Sentence(doc.text)
        tagger.predict(sent1)

        #Parse coreference resolution model response into corefSpans = {text (Span): what its coreferencing (Span)}
        corefSpans={}
        if len(coref['clusters']):
            for index, value in enumerate(coref['predicted_antecedents']):
                if value != -1:
                    source=coref['top_spans'][index]
                    dest=coref['top_spans'][value]
                    corefSpans[doc[source[0]:source[1] + 1]] = doc[dest[0]:dest[1] + 1]
        doc._.DCorefs=corefSpans

        #Disambiguate verb senses - Flair maps to PropBank verb sense. frames = {text (Span): Verb Sense (string)}
        frames={}
        for e in sent1.to_dict(tag_type='frame')['entities']:
            newSpan = (doc.char_span(e['start_pos'],e['end_pos']))
            frames[newSpan] = e['labels'][0].value
        doc._.Uvis = frames

        #Parse Open Information Extraction model response & combine with Verb Sense information.
        oieSubclaims = []
        for e in oie['verbs']:
            oiespans = tags2spans(e['tags'],doc)
            #Drop extracted relations that don't have a verb and at least 1 numbered argument.
            if len(oiespans) < 2 or 'V' not in oiespans or \
                    all(x not in oiespans for x in ['ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4', 'ARG5']) > 0:
                print("axing ",oiespans)
                continue
            oieSubclaims.append(oiespans)

        #Information extraction completed, with doc._.DCorefs storing coreferences, frames storing VSD, oieSubclaims
        #storing extracted relations. These are all shared among the top-level claim, which is now instantiated.
        tlClaim = TLClaim(doc,oieSubclaims)
        #tlClaim.printTL()

#Convert IOB argument tags from the OIE model into spaCy Spans, adjusting for curtailed entities.
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
            while docIn[j].ent_iob_ != 'O':
                j += 1
            spans[spanK] = docIn[span.start:j]

    return spans

if __name__ == '__main__':
    main()



