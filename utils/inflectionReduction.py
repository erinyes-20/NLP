# from util import *

# Add your import statements here
import json
import nltk
from nltk.tokenize import punkt
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.tokenize import RegexpTokenizer	
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')



class InflectionReduction:

	def reduce(self, text):
		"""
		Stemming/Lemmatization

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed/lemmatized tokens representing a sentence
		"""

		wnl = WordNetLemmatizer()

		reducedText = []
		
		for sentence in text:
			sen_lemma=[]
			for token in sentence:
				sen_lemma.append(wnl.lemmatize(token))
				
			reducedText.append(sen_lemma)
			
		#Fill in code here
		
		return reducedText


