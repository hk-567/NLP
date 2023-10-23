# -*- coding: utf-8 -*-
"""NLP.ipynb

"""

#import necessary libraries
import pandas as pd
import numpy as numpy
import nltk #natural language tool kit
nltk.download('popular') #popular packages within NLTK that we need
import nltk.corpus #a corpus can be defined as a collection of text documents

#sample text for tokenization
text = "In Brazil they drive on the right-hand side of the road. Brazil has a large coastline on the eastern side of south america "
#import the tokenize function from nltk
from nltk.tokenize import word_tokenize
#pass string into word tokenize to break the sentence
token = word_tokenize(text)
token

#frequency distinct
from nltk.probability import FreqDist #import FreqDist library
fdist = FreqDist(token)
fdist

fdist1 = fdist.most_common(10)
fdist1




#import portstemmer for nltk library
from nltk.stem import PorterStemmer
pst = PorterStemmer()
print("Porter Stemming example:")
word = "waiting"
stemmed = pst.stem(word)
stemmed

stm = ["give", "gave","giving","given","lazy"]
for word in stm:
  print (word+ ":"+ pst.stem(word))
#wait vs give

#import lancasterstemmer from nltk library
from nltk.stem import LancasterStemmer
lst = LancasterStemmer()

stm = ["give", "gave","giving","given","lazy"]
for word in stm:
  print (word+ ":"+ lst.stem(word))


#import lemmatizer library
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

print("rocks: ", lemmatizer.lemmatize("rocks"))
print("corpora: ", lemmatizer.lemmatize("corpora"))

"""##Stop Words
The most common words in a language like “the”, “a”, “at”, “for”, “above”, “on”, “is”, “all"

>They do not provide any meaning and are usually removed from texts
"""

#import stopwords

#from nltk import word_tokenize
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

tokenizer = ToktokTokenizer()
stopword_list = stopwords.words('english')

a = set(stopwords.words('english'))

text = "Cristiano Ronaldo was born on February 5, 1985, in Funchal, Madeira, Portugal."
def remove_stopwords(text, is_lower_case = False):
  tokens= tokenizer.tokenize(text)
  tokens = [token.strip() for token in tokens]
  if is_lower_case:
    filtered_tokens = [token for token in tokens if token not in stopword_list]
  else:
    filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
  filtered_text = ' '.join(filtered_tokens)
  return filtered_text

remove_stopwords(text)


entities = nltk.chunk.ne_chunk(tagged)
print("entities =", entities)

