# from util import *

# Add your import statements here
import nltk
nltk.download('stopwords') 
from nltk.corpus import stopwords


class StopwordRemoval():

	def fromList(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""
		StopWords = set(stopwords.words('english'))
		stopwordRemovedText = []
		for list in text:
			new_list = [x for x in list if x not in StopWords]
			stopwordRemovedText.append(new_list)

		#Fill in code here

		return stopwordRemovedText




	