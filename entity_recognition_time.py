# time pattern strings.
numbers = "(one|two|three|four|five|six|seven|eight|nine|ten| \
          eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen| \
          eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty| \
          ninety|hundred|thousand)"
week_day = "(monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
month = "(january|february|march|april|may|june|july|august|september| \
          october|november|december)"
dmy = "(year|day|week|month)"
rel_day = "(today|yesterday|tomorrow|tonight|tonite)"
exp1 = "(before|after|earlier|later|ago)"


def entity_recognizer(sentences):
    """
    1. tokenize the sentences in the text.
    2. apply regex pattern matching on the word tokens
    """    
    sentence_lists = nltk.sent_tokenize(sentences) #default sentence segmenter
    word_tokens = [nltk.word_tokenize(sent) for sent in sentence_lists] #word tokenizer
    
    # I dont think we need parts of speech tagger. In case, we need it, we can use this:
    pos_tagged_sents = [nltk.pos_tag(sent) for sent in word_tokens]   