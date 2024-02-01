

# Add your import statements here
# import nltk
from nltk.tokenize import punkt


class SentenceSegmentation:

    def naive(self, text):
        """
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		text : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

        segmented_text = [a.strip(' ') for a in text.replace('? ', '? <>').replace('. ', '. <>').split('<>')]
        if '' in segmented_text:
            segmented_text.remove('')

        # Fill in code here

        return segmented_text

    def punkt(self, text):
        """
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		text : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each strin is a single sentence
		"""
        sent_splitter = punkt.PunktSentenceTokenizer()
        segmented_text = sent_splitter.tokenize(text)

        # Fill in code here

        return segmented_text
