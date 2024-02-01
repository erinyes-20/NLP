

# Add your import statements here
import json
import string
from nltk.tokenize import punkt
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.tokenize import RegexpTokenizer	
from nltk.tokenize import WhitespaceTokenizer


class Tokenization():

    def naive(self, text):
        """
		Tokenization using a Naive Approach

		Parameters
		----------
		text : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""
        pattern = r'\s+'
        regexp = RegexpTokenizer(pattern, gaps=True)
        whitetoken = WhitespaceTokenizer()
        tokenizedText = []
        for a in text:
            tokenizedText.append(regexp.tokenize(a.replace(',', ' , ')))

        # Fill in code here

        return tokenizedText

    def pennTreeBank(self, text):
        """
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		text : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""
        penn = TreebankWordTokenizer()
        tokenizedText = []
        for sentence in text:
            tokens = penn.tokenize(sentence)
            tokens = list(filter(lambda token: token not in string.punctuation, tokens))
            tokenizedText.append(tokens)            

        # Fill in code here

        return tokenizedText
