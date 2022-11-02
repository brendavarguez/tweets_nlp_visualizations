# Libraries
import re
import nltk
import unidecode
import unicodedata
import contractions
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
from googletrans import Translator

class PreProcessor:
    
    def __init__(self, regex_dict = None):
        
        # creating classes
        # stem
        self.sb = nltk.stem.SnowballStemmer('english')
        
        # lemmatize
        self.lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
        
        # translate
        self.translator = Translator()
        
        # declare a default regex dict
        self.default_regex_dict = {'goo[o]*d':'good', '2morrow':'tomorrow', 'b4':'before', 'otw':'on the way',
                                   'idk':"i don't know", ':)':'smile', 'bc':'because', '2nite':'tonight',
                                   'yeah':'yes', 'yeshhhhhhhh':'yes', ' yeeeee':'yes', 'btw':'by the way', 
                                   'fyi':'for your information', 'gr8':'great', 'asap':'as soon as possible', 
                                   'yummmmmy':'yummy', 'gf':'girlfriend', 'thx':'thanks','nowwwwwww':'now', 
                                   ' ppl ':' people ', 'yeiii':'yes'}
        
        # if no regex_dict defined by user, then use 
        # one by default. Else, concat two regex dicts
        if regex_dict:            
            self.regex_dict = {**regex_dict, **default_regex_dict}
            
        else:
            self.regex_dict = self.default_regex_dict
    
    def translate_twt(self, pdf):
        """
        This function helps to translate a tweet from any 
        language to English.

        Inputs:
            - pdf: Pandas dataframe. This dataframe must have
               the following columns:
                - lang: Tweet's language.
                - clean_tweet: Partially pre-processed tweet.

        Outputs: Translated tweet from any language available 
                 in googletrans api to English.
        """

        # Check if the language of the tweet is either undefined or English
        # to avoid translation.
        if pdf["lang"] == "und" or pdf["lang"] == "en":
            pdf["translated_tweet"] = pdf["clean_tweet"]

        # Check if tweet is in Hindi. The code of Hindi language is "hi", but 
        # Twitter has defined the code as "in".
        elif pdf["lang"] == "in":
            pdf["translated_tweet"] = self.translator.translate(pdf["clean_tweet"], src = "hi", dest = "en").text
            
        # Check if tweet is in Chinese. 
        # The api supports simplified and traditional chinese.
        elif pdf["lang"] == "zh":
            pdf["translated_tweet"] = self.translator.translate(pdf["clean_tweet"], src = "zh-cn", dest = "en").text

        # For any other language the translator should work just fine, so the
        # api should work with the language detected by Twitter.
        else:
            try:
                pdf["translated_tweet"] = self.translator.translate(pdf["clean_tweet"], src = pdf["lang"], 
                                                                    dest = "en").text
            except (TypeError, ValueError):
                pdf["translated_tweet"] = pdf["clean_tweet"]
                
        return pdf["translated_tweet"]

    
    def removeNoise(self, pdf):
        """
        Function to remove noise from strings. 
        
        Inputs: A pandas dataframe with raw strings of length n.
        
        Output: A clean string where elements such as accented 
        words, html tags, punctuation marks, and extra white 
        spaces will be removed (or transform) if it's the case.
        """
        
        # to lower case
        pdf["clean_tweet"] = pdf.text.apply(lambda x: x.lower())
        
        # remove accented characters from string
        # e.g. canción --> cancion
        pdf["clean_tweet"] = pdf.clean_tweet.apply(lambda x: unidecode.unidecode(x))
        
        # remove html tags 
        pdf["clean_tweet"] = pdf.clean_tweet.str.replace(r'<[^<>]*>', '', regex = True)
        
        # remove (match with) usernames | hashtags | punct marks | links
        # punct marks = ",.':!?;
        # do not remove: ' 
        # but remove: "
        pdf["clean_tweet"] = pdf.clean_tweet.apply(lambda x:' '.join(re.sub("(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)|([-.,:_;])|(https?:\/\/.*[\r\n]*)",
                                                                            "", x).split()).replace('"',''))
                
        # remove white spaces at the begining and at 
        # the end of a string
        pdf['clean_tweet'] = pdf.clean_tweet.apply(lambda x: x.lstrip(' '))
        pdf['clean_tweet'] = pdf.clean_tweet.apply(lambda x: x.rstrip(' '))
        
        # Translate tweet
        pdf["clean_tweet"] = pdf.apply(lambda x: self.translate_twt(x) \
                                       if pd.isnull(x.clean_tweet) == False else x, axis = 1)   

        # normalize string
        # normalize accented charcaters and other strange characters
        # NFKD if there are accented characters (????
        pdf["clean_tweet"] = pdf.clean_tweet.apply(lambda x: unicodedata.normalize('NFKC', x)\
                                                   .encode('ASCII', 'ignore').decode("utf-8")\
                                                   if (pd.isnull(x) == False and x != "") else x)
        
        return pdf
    
    
    def textNormalization(self, pdf):
        """
        Function to normalize a string. 
        
        Inputs: A pandas dataframe with strings (of length n) that 
        will be normalized. 
        
        Outputs: A normalized string whitout noise, words in their
        (expected) correct form and with no stopwords.
        """
        
        # remove noise first
        pdf = self.removeNoise(pdf)

        # expand contractions
        # e.g. don't --> do not
        pdf['clean_tweet'] = pdf.clean_tweet.apply(lambda x: contractions.fix(x))
 
        # Normalize words
        pdf['clean_tweet'] = pdf.clean_tweet.replace(self.regex_dict)
                
        # get English stopwords    
        stop_words = stopwords.words('english')
        stopwords_dict = Counter(stop_words)
        
        # remove stopwords from string
        pdf["clean_tweet"] = pdf.clean_tweet.apply(lambda x: ' '.join([word for word in x.split()
                                                                       if word not in stopwords_dict]))
            
        return pdf
    
    
    def wordTokenize(self, pdf):
        """
        Function to tokenize a string into words. Tokenization is a way 
        of separating a piece of text into smaller units called tokens.
        In this case tokens are words (but can also be characters or 
        subwords).
        
        Inputs: A pandas dataframe with strings (of length n) that will be tokenized. 
        
        Outputs: A list of tokenized words.
        """
        # string normalized
        pdf = self.textNormalization(pdf)
        
        # Use word_tokenize method to split the string
        # into individual words. By default it returns
        # a list.
        pdf["clean_tweet"] = pdf.clean_tweet.apply(lambda x: nltk.word_tokenize(x))        
        
        # Using isalpha() will help us to only keep
        # items from the alphabet (no punctuation
        # marks). 
        #pdf["clean_tweet"] = pdf.clean_tweet.apply(lambda x: [word for word in x if word.isalpha()])
        
        # Keep only unique elements
        pdf["clean_tweet"] = pdf.clean_tweet.apply(lambda x: list(set(x)))

        # return list of tokenized words by row
        return pdf
    
    def phraseTokenize(self, pdf):
        """
        Function to tokenize a string into sentences. Tokenization is
        a way of separating a piece of text into smaller units called
        tokens. In this case tokens are phrases (but can also be words,
        characters or subwords).
        
        Inputs: A string (of length n) that will be tokenized. 
        
        Outputs: A list of tokenized sentences.
        """
        
        # pandas dataframe with strings normalized
        pdf = self.textNormalization(pdf)
        
        # Use sent_tokenize method to split the string
        # into sentences. By default it returns a list.
        pdf["clean_tweet"] = pdf.clean_tweet.apply(lambda x: nltk.sent_tokenize(x))   
        
        return pdf 
    
    
    def stemWords(self, pdf):
        """
        Function to stem strings. Stemming is the process of reducing
        a word to its word stem that affixes to suffixes and prefixes 
        or to the roots of words (known as a lemma).
        
        Inputs: A raw string of length n.
        
        Output: Roots of each word of a given string.
        """
        
        # pandas dataframe with strings normalized
        pdf = self.textNormalization(pdf)
        
        # tokenized string (into words)
        pdf = self.wordTokenize(data)
            
        # reduct words to its root    
        pdf["clean_tweet"] = pdf.clean_tweet.apply(lambda x: [self.sb.stem(word) for word in x])
        
        return pdf
    
    
    def lemmatizeWords(self, pdf):
        """
        Function to lemmatize strings. Lemmatization is a method 
        responsible for grouping different inflected forms of 
        words into the root form, having the same meaning. It is 
        similar to stemming.
        
        Inputs: A raw string of length n.
        
        Output: Roots of each word of a given string (with better
        performance than in stemming).
        """
        unw_chars = ["(", ")", "[", "]"]
        
        # pandas dataframe with strings normalized
        pdf = self.textNormalization(pdf)
        
        # list of tokenized words (from string)
        # Here it was decided to tokenize by words
        # rather than by sentences due to it might
        # be easier to find the correct roots
        # of each word
        pdf = self.wordTokenize(pdf)
        
        # lematize word from list of tokenized words
        #lematized = [self.lemmatizer.lemmatize(word) for word in tokenized]
        pdf["clean_tweet"] = pdf.clean_tweet.apply(lambda x: [self.lemmatizer.lemmatize(word) 
                                                              for word in x if word not in unw_chars])
        
        return pdf