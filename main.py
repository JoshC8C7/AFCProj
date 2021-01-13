from allennlp.predictors.predictor import Predictor
import pandas as pd
import os
import spacy
from spacy.tokens import Doc, Span
from flair.data import Sentence
from flair.models import SequenceTagger
from claim import TLClaim, getCorefs, argIDGen
from spacy import displacy
import connectives as con
import pathlib


def main():
    #spacy.require_gpu()
    pathlib.PosixPath = pathlib.WindowsPath #Small hack to handle AllenNLP & Flair not being compatible with windows
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
        if (True or 'safely' in s) and any(x !=" " for x in s):
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
        """for ik,iv in corefSpans.items():
            print(ik.start, ik.text, iv.text)
        print(corefSpans)"""
        doc._.DCorefs=corefSpans

        frames={}
        for e in sent1.to_dict(tag_type='frame')['entities']:
            #print(e['text'], " ",e['start_pos']," -> ", e['end_pos'], ":",e['labels'][0].value)
            newSpan = (doc.char_span(e['start_pos'],e['end_pos']))
            frames[newSpan] = e['labels'][0].value


        oieSubclaims = []
        uviDict = {}
        for e in oie['verbs']:
            oiespans = tags2spans(e['tags'],doc)
            if len(oiespans) < 2 or 'V' not in oiespans or all(x not in oiespans for x in ['ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4', 'ARG5']) > 0:
                print("axing ",oiespans)
                continue
            oieSubclaims.append(oiespans)
            for key,val in oiespans.items():
                if key == 'V':
                    uvi = frames.get(val,None)
                    if uvi is not None:
                        uviDict[argIDGen(val)] = uvi

        tlClaim = TLClaim(doc,oieSubclaims,uviDict)
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
    pre = spans.copy()

    for spanK, span in spans.items():
        if docIn[span.start].ent_iob_ == 'I':
            i=span.start-1
            while docIn[i].ent_iob_ != 'B':
                i-=1
            spans[spanK] = docIn[i:span.end]

    #This has to be two loops as the iterable is modified, so need it written back before changing the start index.
    for spanK, span in spans.items():
        if docIn[span.end-1].ent_iob_ in ['I','B'] and span.end+1 < len(docIn):
            j = span.end
            while docIn[j].ent_iob_ != 'O':
                j += 1
            spans[spanK] = docIn[span.start:j]

    """if pre !=spans:
        print()
        print("SSSSSSSSSSSSSSS")
        print("ents ", docIn.ents)
        print("nc ", list(docIn.noun_chunks))
        print("Pre ", pre)
        print("Post ", spans)
        print("EEEEEEEEEEEEEEEE")
        print()"""

    return spans

if __name__ == '__main__':
    main()



