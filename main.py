from allennlp.predictors.predictor import Predictor
from pprint import pprint
import pandas as pd
import os
import spacy
from spacy.tokens import Doc, Span
from flair.data import Sentence
from flair.models import SequenceTagger
from nltk.corpus import propbank
from claim import TLClaim
from claim import Claim
from claim import getCorefs
import nltk.sem
from spacy import displacy
import connectives as con

def main():
    os.environ["CORENLP_HOME"] = "~/stanza_corenlp"
    nlp=spacy.load('en_core_web_lg')
    Doc.set_extension("DCorefs",default={})
    Doc.set_extension("ConnectiveEdges", default=[])
    Span.set_extension("SCorefs",getter=getCorefs)
    nlp.add_pipe(con.extractConnectives,name='extract_connectives',last=True)

    df=pd.read_table(os.path.join("data","Politihop","Politihop_train.tsv"),sep='\t')
    statementSet = set()

    for s in df['statement']:
        while not s[0].isalpha():
            s=s[1:]
        if s.split(" ")[0].lower() == "says":
            s=(s.partition(" "))[1]
        s.replace('&amp;','and')
        if False or 'TV' in s:
            statementSet.add(s)

    predictorOIE = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz")
    predictorCoref = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz")
    tagger = SequenceTagger.load('frame')
    print("Models Loaded")

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
        print(corefSpans)
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

