import pandas as pd
from os import path
import evidence

from nlpPipeline import batch_proc
from claim import DocClaim
from web_scrape import nlp_feed

# Politihop-to-standard label conversion, as in Politihop.
politiDict = {'true': 1, 'mostly-true': 1, 'barely-true': -1, 'half-true': 0, 'mostly-false': -1, 'pants-fire': -1,
              'false': -1}


# Handles input for politihop
def politihop_input(data):
    # Read in input claims
    df = pd.read_table(path.join("data", "Politihop", data), sep='\t').head(100)

    # The input claims data has multiple repetitions of each text due to containing multiple verifiable claims. This
    # is handled later so for now the text must be de-duplicated. Other text pre-processing/cleansing occurs here.
    statement_set = set()
    truth_dict = {}
    for i, row in df.iterrows():
        statement = row['statement']
        truth = row['politifact_label']
        while not statement[0].isalpha() or statement[0] == " ":
            statement = statement[1:]

        # Push in name of author to claim where a real person i.e. not a viral image/post
        if statement.partition(" ")[0].lower() == "says":
            author = row['author'].replace("Speaker: ", "")
            if True or author in ['Facebook posts', 'Viral image']:
                statement = statement.partition(" ")[2]
            else:
                statement = author + " s" + statement[1:]

        if statement:
            statement_set.add(statement)
        truth_dict[statement] = politiDict[truth]
    return statement_set, truth_dict


def liar_input(data):
    # Read in input claims
    df = pd.read_table(path.join("data", "liarliar", data), sep='\t').head(200)

    # The input claims data has multiple repetitions of each text due to containing multiple verifiable claims. This
    # is handled later so for now the text must be de-duplicated. Other text pre-processing/cleansing occurs here.
    statement_set = set()
    truth_dict = {}
    for i, row in df.iterrows():
        statement = row[2]
        truth = row[1]
        while not statement[0].isalpha() or statement[0] == " ":
            statement = statement[1:]
        if statement.partition(" ")[0].lower() == "says":
            statement = statement.partition(" ")[2]
        statement_set.add(statement)
        truth_dict[statement] = politiDict[truth]

    return statement_set, truth_dict


# Change which dataset to import
DATA_IMPORT = {'politihop': politihop_input, 'liarliar': liar_input}


# Run inference on a single claim
def process_claim(doc, truth_dict, limiter):
    # Setup evidence store & Per subclaim results
    claim_ev = []
    print("CLAIM: ", doc)
    sc_level_results = []

    # Split claim into subclaim and then logical formulae
    top_level_claim = DocClaim(doc)

    # Iterate through generated subclaims and attempt to prove.
    for subclaim in top_level_claim.subclaims:

        # Obtain search terms from processed subclaims
        query, noun_ch, entities = subclaim.kb.prep_search()

        # Collect evidence from webcrawler, modified by evidence domain limiter as appropriate.
        sources = []
        if (limiter is not None) or not query:
            sources.extend(nlp_feed(top_level_claim.doc.text,limiter))
        elif query:
            sources.extend(nlp_feed(query))


        # Process collected evidence into knowledge base.
        evidence.process_evidence(subclaim, noun_ch, entities, sources)

        # Attempt proof
        result, sub_claim_evidence = subclaim.kb.prove()
        claim_ev.append(sub_claim_evidence)

        # Write result back, 1 if true, -1 if false/no proof found*, or error symbol '5'.
        if result is not None:
            sc_level_results.append(1 if result else -1)
        else:
            sc_level_results.append(5)

    # Determine overall document result from subclaim results.
    if sc_level_results:
        print("Subclaim Results: ", sc_level_results)
        proportion = max(sc_level_results)
        print("Guessed Truth:", str(proportion), "  Ground Truth:", truth_dict[doc.text])
        return [proportion, truth_dict[doc.text], claim_ev]
    else:
        return [5, truth_dict[doc.text], []]


def main(name='politihop', data='politihop_train.tsv', limiter=None):
    results = []
    # Read in statements & associated Ground truth
    statement_set, truth_dict = DATA_IMPORT[name](data)
    length = len(statement_set)

    # Batch-process spaCy on documents
    docs = batch_proc(statement_set)

    # Convert doc to claim, run inference, append results to list
    for index, doc in enumerate(docs):
        print("Processing Doc: ", index+1, " of ", length)
        res = process_claim(doc, truth_dict, limiter)
        print(res)
        results.append([doc, res[0], res[1], res[2]])

    # Return final results
    print(results)
    import csv
    # opening the csv file in 'w+' mode
    file = open('output.csv', 'w+', newline='')

    # writing the data into the file
    with file:
        write = csv.writer(file)
        write.writerows(results)


if __name__ == '__main__':
    main('politihop','politihop_train.tsv','pfOnly')
