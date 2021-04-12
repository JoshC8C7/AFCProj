import spacy
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span
import torch
import connectives as con
from pprint import pprint
import textacy.similarity as ts
from transformer_srl import dataset_readers,models,predictors
import  neuralcoref
import Levenshtein
countX=0
countY=0
#This is a module rather than a class to enforce singleton behaviour of the models. The nlp model is used across the
#files and so should be accessible from each module rather than through a single object.
BATCH_SIZE = 100
def docUsefullness(doc,miniContext):
    #miniContext = (ncs, entities)
    #print(doc)

    for i in doc.ents:
        for j in miniContext[1]:
            threshold = 0.5 if i.label_ == j.label_ else 0.7
            if ts.token_sort_ratio(i.lower_,j.lower_) > threshold:# or i.similarity(j) > threshold:
                #print("Proceeding to OIE")
                global countX
                countX+=1
                return True

    leftoversExisting = list(x for x in miniContext[0] if x not in miniContext[1])
    leftoversIncoming = list(x for x in doc.noun_chunks if x not in doc.ents)

    for i in leftoversIncoming:
        for j in leftoversExisting:
            if i.similarity(j) > 0.8 or Levenshtein.ratio(i.lower_,j.lower_) > 0.6:
                #print("Proceeding to OIE")
                global countY
                countY+=1
                return True
    return False

def oieParse(oie, doc):
    # Parse Open Information Extraction model response & combine with Verb Sense information.
    oieSubclaims = []
    frames={}
    for e in oie['verbs']:
        oiespans = tags2spans(e['tags'], doc)
        # Drop extracted relations that don't have a verb and at least 1 numbered argument.
        if len(oiespans) < 2 or 'V' not in oiespans or \
                all(x not in oiespans for x in ['ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4', 'ARG5']) > 0:
            #print("axing ", oiespans)
            continue
        #print(oiespans['V'], " ",e['frame'], " ", e['frame_score'])
        frames[oiespans['V']] = e['frame']
        oieSubclaims.append(oiespans)

    #Store OIEs with the document. Whilst these are only retrieved once, they must be stored within the document
    #to allow spaCy to parallelize.
    doc._.OIEs = oieSubclaims
    doc._.Uvis = frames

    return doc

#todo move this to be inside spacy's pipeline possibly
def oieBatch(docs):
    """newDocs = []
    for doc in docs:
        newDocs.append(oiePipe(doc))"""

    #Allennlp is slightly incomplete in that there's no batch text->prediction functionality, so
    #instead a small hack to impersonate a json:
    #todo  possibly fix allennlp if the licence allows it so it doesn't drop all the token info and then just retokenize everything.

    batched_jsons = []
    oies=[]
    index = 0

    while index < len(docs):
        batched_jsons.append(list({'sentence':x.text} for x in docs[index:min(index+BATCH_SIZE,len(docs))]))
        index+=BATCH_SIZE

    for i in batched_jsons:
        #print(i)
        oies.extend(predictorOIE.predict_batch_json(i))
    newDocs = (list(map(oieParse,oies,docs)))

    return newDocs

def oiePipe(doc):
    #print("PIPING: ", doc)
    oie = predictorOIE.predict_tokenized(list(tok.text for tok in doc))
    doc = oieParse(oie,doc)
    return doc



# Load in spaCy (Tokenizer, Dep parse, NER), and set extensions required for later use.
print("Initiating model load...",end="")
nlp = spacy.load('en_core_web_lg')
print("..")
Doc.set_extension("Uvis", default={})
Doc.set_extension("OIEs", default={})
Doc.set_extension("ConnectiveEdges", default=[])
Doc.set_extension("rootDate",default='')
Doc.set_extension("url",default='')
# Run connective extractor over input text and store result in doc._.extract_connectives.
coref = neuralcoref.NeuralCoref(nlp.vocab)
nlp.add_pipe(coref, name='neuralcoref')
nlp.add_pipe(oiePipe,name='oiePipe',last=True)
nlp.add_pipe(con.extractConnectives, name='extract_connectives', last=True)

matcher = Matcher(nlp.vocab)
matcher.add("quotes",[[{'ORTH': '"'},{'IS_ASCII': True, 'OP': '*'}]])

predictorOIE = predictors.SrlTransformersPredictor.from_path("data/srl_bert_base_conll2012.tar.gz", "transformer_srl", cuda_device=0, language='en_core_web_lg')
print("Models Loaded")

def naiveQuotes(doc):
    matches=matcher(doc)
    if matches:
        #print("DROP:", doc)
        return True
    else:
        return False


def batchProc(statements, dateMap, urlMap=None, miniContext = None):
    #See 25 - http://assets.datacamp.com/production/course_8392/slides/chapter3.pdf
    if urlMap is None: urlMap = {}
    docsOut = []
    inputData = list(zip(statements,({'date':dateMap.get(s,None), 'url':urlMap.get(s,None)} for s in statements)))
    #print(len(inputData))
    if miniContext is not None:
        with nlp.disable_pipes(['extract_connectives','oiePipe']):
            doc_pool = []
            for doc, context in nlp.pipe(inputData,as_tuples=True):
                doc._.rootDate = context['date']
                doc._.url = context['url']
                if docUsefullness(doc,miniContext) and not naiveQuotes(doc) and len(doc) < 500:
                    doc_pool.append(doc)
            docsOut=oieBatch(doc_pool)
    else:
        for doc, context in nlp.pipe(inputData, as_tuples=True):
            doc._.rootDate = context['date']
            docsOut.append(doc)
    print(" /////////////////////////////////////////////////////////////////////////////")
    print("COUNTx:",countX)
    print("COUNTy:",countY)
    return docsOut

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