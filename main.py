from allennlp.predictors.predictor import Predictor
from pprint import pprint
import pandas as pd
import os
import spacy
from spacy.tokens import Doc, Span
from flair.data import Sentence
from flair.models import SequenceTagger
from nltk.corpus import propbank
import xml
from claim import TLClaim
from claim import Claim


#Method to handle coreferences
#If there is a registered coreference (as set under doc._.corefs) then check if its within the current span/argument,
#if there is one then return that (i.e. return 'Bob' rather than 'he'), otherwise just return the span
def getCoref(span):
    for corefK, corefV in span.doc._.DCorefs.items():
        #is coref within span?
        if corefK.start >= span.start and corefK.end <=span.end:
            #Matching on the first found is sound as all matches (coref->antecedent) are injective.
            if not any(tok.pos_ == "PRON" for tok in corefV):
                return corefV
    #No matches, so just return the span for onwards use.
    return span

def main():
    nlp=spacy.load('en_core_web_lg')
    Doc.set_extension("DCorefs",default={})
    Span.set_extension("SCorefs",getter=getCoref)

    df=pd.read_table(os.path.join("data","Politihop","Politihop_train.tsv"),sep='\t')
    statementSet = set()

    for s in df['statement']:
        while not s[0].isalpha():
            s=s[1:]
        if s.split(" ")[0].lower() == "says":
            s=(s.partition(" "))[1]
        statementSet.add(s)

    predictorOIE = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz")
    predictorCoref = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz")
    tagger = SequenceTagger.load('frame')

    for s in statementSet:
        doc=nlp(s)
        oie = predictorOIE.predict(s)
        coref = predictorCoref.predict(s)
        sent1=Sentence(doc.text)
        tagger.predict(sent1)

        corefSpans={}
        if len(coref['clusters']):
            for index, value in enumerate(coref['predicted_antecedents']):
                if value != -1:
                    source=coref['top_spans'][index]
                    dest=coref['top_spans'][value]
                    corefSpans[doc[source[0]:source[1] + 1]] = doc[dest[0]:dest[1] + 1]
        doc._.DCorefs=corefSpans

        frames={}
        for e in sent1.to_dict(tag_type='frame')['entities']:
            #print(e['text'], " ",e['start_pos']," -> ", e['end_pos'], ":",e['labels'][0].value)
            newSpan = (doc.char_span(e['start_pos'],e['end_pos']))
            frames[newSpan] = e['labels'][0].value


        subclaims = []
        for e in oie['verbs']:
            oiespans = tags2spans(e['tags'],doc)
            for key,val in oiespans.items():
                if key == 'V':
                    uvi = frames.get(val,None)
                    if uvi is not None:
                        newClaim = Claim(doc, oiespans, (val,uvi))
                        subclaims.append(newClaim)

        tlClaim = TLClaim(doc,subclaims)
        tlClaim.generateCG()
        #tlClaim.printTL()

def tags2spans(tags,docIn):
    spans={}
    start=0
    open=False
    for index,tag in enumerate(tags):
        if open:
            if tag[0] in ('O','B') or index==len(tags)-1:
                #If hit an empty and prevous still going, need to close it and write.
                spans[tags[index - 1][2:]] = docIn[start:index]
                if tag[0] == 'O':
                    open=False

        if tag[0] == 'B':
            # Open new entry and set start index accordingly.
            start = index
            open = True
    return spans

if __name__ == '__main__':
    main()

