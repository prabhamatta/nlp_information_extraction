import nltk

# time pattern strings.
numbers = "(one|two|three|four|five|six|seven|eight|nine|ten| \
          eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen| \
          eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty| \
          ninety|hundred|thousand)"
week_day = "(monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
month = "(january|february|march|april|may|june|july|august|september| \
          october|november|december)"
dmy = "(year|day|week|month)"
relation_day = "(today|yesterday|tomorrow|tonight|tonite)"
phrasal_words = "(before|after|earlier|later|ago)"

def sub_pattern_matcher(sent):
    matched_patterns = []
     #if pattern matches, add it to a list --> matched_patterns
    #return matched_patterns
    return sent


def super_pattern_matcher(sent):
    matched_patterns = []
     #if pattern matches, add it to a list --> matched_patterns
    #return matched_patterns
    return sent

    
def entity_recognizer(sentences):
    """
    1. tokenize the sentences in the text.
    2. apply regex pattern matching on the word tokens
    """    
    sentence_lists = nltk.sent_tokenize(sentences) #default sentence segmenter
    sent_tokens = [nltk.word_tokenize(sent) for sent in sentence_lists] #word tokenizer
    print "\nsentence tokens===",sent_tokens    
    
    # I dont think we need parts of speech tagger. In case, we need it, we can use this:
    pos_tagged_sents = [nltk.pos_tag(sent) for sent in sent_tokens]  
    print "\npost_tagger==",pos_tagged_sents
    
    
    #make into lowercase
    lowercase_sent_tokens = []
    for s_tokens in sent_tokens:
        lowercase_tokens = []
        for token in s_tokens:
            lowercase_tokens.append(token.lower())
        lowercase_sent_tokens.append(lowercase_tokens)
        
    print "\nlowercase===",lowercase_sent_tokens
        
    
    #call pattern matching here to find all time-related patterns"     
    all_time_patterns = super_pattern_matcher(lowercase_tokens)
    
    for phrase in all_time_patterns:
        #  call sub pattern matching for sub patterns
        sub_patterns = sub_pattern_matcher(phrase)
        #print sub_patterns
    return         

if __name__ == '__main__':
    text = "Today is Thursday. Tomorrow will be beautiful"
    entity_recognizer(text)
