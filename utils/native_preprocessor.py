
from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval
# from informationRetrieval import InformationRetrieval
# from evaluation import Evaluation

from sys import version_info
import argparse
import json
import matplotlib.pyplot as plt

class native_prep():
    
    ### Query preproc
    def preprocessQueries(self, queries):
            """
            Preprocess the queries - segment, tokenize, stem/lemmatize and remove stopwords
            """
            tokenizer = Tokenization()
            sentenceSegmenter = SentenceSegmentation()
            inflectionReducer = InflectionReduction()
            stopwordRemover = StopwordRemoval()

            # Segment queries
            segmentedQueries = []
            for query in queries:
                segmentedQuery = sentenceSegmenter.punkt(query)
                segmentedQueries.append(segmentedQuery)
            # json.dump(segmentedQueries, open(self.args.out_folder + "segmented_queries.txt", 'w'))
            # Tokenize queries
            tokenizedQueries = []
            for query in segmentedQueries:
                tokenizedQuery = tokenizer.pennTreeBank(query)
                tokenizedQueries.append(tokenizedQuery)
            # json.dump(tokenizedQueries, open(self.args.out_folder + "tokenized_queries.txt", 'w'))
            # Stem/Lemmatize queries
            reducedQueries = []
            for query in tokenizedQueries:
                reducedQuery = inflectionReducer.reduce(query)
                reducedQueries.append(reducedQuery)
            # json.dump(reducedQueries, open(self.args.out_folder + "reduced_queries.txt", 'w'))
            # Remove stopwords from queries
            stopwordRemovedQueries = []
            for query in reducedQueries:
                stopwordRemovedQuery = stopwordRemover.fromList(query)
                stopwordRemovedQueries.append(stopwordRemovedQuery)
            # json.dump(stopwordRemovedQueries, open(self.args.out_folder + "stopword_removed_queries.txt", 'w'))

            preprocessedQueries = stopwordRemovedQueries
            return preprocessedQueries


    ### Doc preproc
    def preprocessDocs(self, docs):
            """
            Preprocess the documents
            """
            tokenizer = Tokenization()
            sentenceSegmenter = SentenceSegmentation()
            inflectionReducer = InflectionReduction()
            stopwordRemover = StopwordRemoval()

            # Segment docs
            segmentedDocs = []
            for doc in docs:
                segmentedDoc = sentenceSegmenter.punkt(doc)
                segmentedDocs.append(segmentedDoc)
            # json.dump(segmentedDocs, open(self.args.out_folder + "segmented_docs.txt", 'w'))
            # Tokenize docs
            tokenizedDocs = []
            for doc in segmentedDocs:
                tokenizedDoc = tokenizer.pennTreeBank(doc)
                tokenizedDocs.append(tokenizedDoc)
            # json.dump(tokenizedDocs, open(self.args.out_folder + "tokenized_docs.txt", 'w'))
            # Stem/Lemmatize docs
            reducedDocs = []
            for doc in tokenizedDocs:
                reducedDoc = inflectionReducer.reduce(doc)
                reducedDocs.append(reducedDoc)
            # json.dump(reducedDocs, open(self.args.out_folder + "reduced_docs.txt", 'w'))
            # Remove stopwords from docs
            stopwordRemovedDocs = []
            for doc in reducedDocs:
                stopwordRemovedDoc = stopwordRemover.fromList(doc)
                stopwordRemovedDocs.append(stopwordRemovedDoc)
            # json.dump(stopwordRemovedDocs, open(self.args.out_folder + "stopword_removed_docs.txt", 'w'))

            preprocessedDocs = stopwordRemovedDocs
            return preprocessedDocs
