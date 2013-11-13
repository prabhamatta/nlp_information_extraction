__authors__ = 'Prabhavathi Matta, Jason Blum'
__Project__= 'Develop TIME Entity Recognition algorithm, Develop TIME_PHRASE chunker, Customized funtions to caluculate Precision and Recall'
import nltk
import re
from pprint import pprint
import os


"""
********************************************************************
Project Members:

a) Prabhavathi Matta - wrote patterns for TIME Entity recognition, worked on TIME_PHRASE parser , generated labelled data for testing, wrote functions to test Precision and Recall of our TIME entity algorithm, and code cleanup & documentation.

b) Jason Blum - wrote funtions for TIME entity pattern matching, worked on TIME_PHRASE parser
********************************************************************


PROJECT:
In this project, we tried to achieve 3 major tasks:
********************************************************************
1. Developed an  Entity Recognition Algorithm to identify TIME Entity.
********************************************************************
using regex pattern matching we match the text for days, months, years, phrasal words like "before|after|earlier|later|ago", etc (more information given below). Once we match these words, we add the word "TIME" to the original POS of the text. This results in different types  of tags like NN_TIME, CD_TIME, NNP_TIME, JJ_TIME, etc



For the sample text given below (TEST_TEXT), TIME tags are labelled to the sentences:

[[('I', 'PRP'),
  ('went', 'VBD'),
  ('to', 'TO'),
  ('the', 'DT'),
  ('store', 'NN'),
  ('this', 'DT'),
  ('morning', 'NN_TIME'),
  ('.', '.')],
[('Early', 'RB'),
  ('last', 'JJ'),
  ('night', 'NN_TIME'),
  ('I', 'PRP'),
  ('went', 'VBD'),
  ('to', 'TO'),
  ('the', 'DT'),
  ('building', 'NN'),
  ('.', '.')],
[('The', 'DT'),
  ('Neolithic', 'NNP'),
  ('age', 'NN'),
  ('in', 'IN'),
  ('China', 'NNP'),
  ('can', 'MD'),
  ('be', 'VB'),
  ('traced', 'VBN'),
  ('back', 'RP'),
  ('to', 'TO'),
  ('about', 'IN'),
  ('5,0000BC', 'CD'),
  ('.', '.')],
  .
  .
  .
  .
[('Let', 'NNP'),
  ('us', 'PRP'),
  ('meet', 'VBP'),
  ('tomorrow', 'NN_TIME'),
  ('at', 'IN'),
  ('2pm', 'CD_TIME'),
  ('.', '.')],
 [('Shall', 'NNP'),
  ('we', 'PRP'),
  ('go', 'VBP'),
  ('at', 'IN'),
  ('5:00pm', 'CD_TIME'),
  ('?', '.')]]
 
 
Further, a summary of tags for the whole blob of text is generated as follows:

[['morning'],
 [],
 [],
 ['night'],
 ['years', 'ago'],
 ['years', 'ago'],
 ['later'],
 ['1923-27'],
 [],
 ['7000', 'BC'],
 ['7000', '5800BC'],
 ['6000-5000', 'BC'],
 ['tomorrow', '2pm'],
 ['5:00pm'],
 ['1000']]





********************************************************************
2. Tested Precision and Recall of our TIME entity algorithm. 
********************************************************************
Since TIME Entity is not present in nltk chunker, we had to develop our own labelled data and test our algorithm against it.

format of the labelled data:
=============
<sent1> <list of time tags in sent1>
<sent2> <list of time tags in sent1>
<sent3> <list of time tags in sent1>

For eg:
Her birthday is on Friday	Friday
Let us meet tomorrow at 2pm.	tomorrow,2pm

***************************
RESULTS:
Num of items tested on == 114
Precision == 0.857142857143
Recall== 0.8
***************************

Sample detailed output:

SENTENCES                                                                                                    CORRECT TIME TAGS       PREDICTED TIME TAGS
**************************************************                                                             ***************           ***************

['I', 'went', 'to', 'the', 'store', 'this', 'morning', '.']                                                        ['morning']               ['morning']
['I', 'went', 'to', 'the', 'store', 'this', 'morning', ',', 'however', 'it', 'is', 'late', '10am', '.']       ['morning', '10am']       ['morning', '10am']
['I', 'went', 'to', 'the', 'building', '.']                                                                                 []                        []
['Early', 'last', 'night', 'I', 'went', 'to', 'the', 'building', '.']                                                ['night']                 ['night']
['Farming', 'gave', 'rise', 'to', 'the', 'Jiahu', 'culture', 'in', '5800BC', '.']                                   ['5800BC']                ['5800BC']
['Meet', 'me', 'on', '11/22']                                                                                        ['11/22']                        []
['I', 'will', 'come', 'to', 'the', 'USA', 'on', '15th', 'March']                                             ['15th', 'March']                 ['March']
['Her', 'birthday', 'is', 'on', 'Friday']                                                                           ['Friday']                ['Friday']




********************************************************************
3. Developed an algorithm for TIME_PHRASE chunker 
********************************************************************

We used the TIME tags developed in (1) to generate TIME_PHRASE chunker using the regular expression chunker nltk.RegexpParser. 

INPUT:I went to the store this morning.
OUTPUT:
(S
  I/PRP
  went/VBD
  to/TO
  (NP the/DT store/NN)
  (TIME_PHRASE this/DT morning/NN_TIME)
  ./.)

INPUT:Let us meet tomorrow at 2pm.
OUTPUT:
(S
  (NP Let/NNP)
  us/PRP
  meet/VBP
  (TIME_PHRASE tomorrow/NN_TIME)
  (TIME_PHRASE at/IN 2pm/CD_TIME)
  ./.)

INPUT: Shall we go at 5:00pm?
OUTPUT:
 (S
  (NP Shall/NNP)
  we/PRP
  go/VBP
  (TIME_PHRASE at/IN 5:00pm/CD_TIME)
  ?/.)


********************************************************************
4. Summary of all TIME_PHRASES in a blob of text.
********************************************************************

 For the sample text given below (TEST_TEXT), OUTPUT is:
    
    (TIME_PHRASE this/DT morning/NN_TIME)
    (TIME_PHRASE Early/RB last/JJ night/NN_TIME)
    (TIME_PHRASE years/NNS_TIME ago/IN_TIME)
    (TIME_PHRASE million/CD years/NNS_TIME ago/IN_TIME)
    (TIME_PHRASE later/JJ_TIME)
    (TIME_PHRASE in/IN 1923-27/CD_TIME)
    (TIME_PHRASE about/IN 7000/CD_TIME BC/NNP_TIME)
    (TIME_PHRASE 7000/CD_TIME)
    (TIME_PHRASE 5800BC/CD_TIME)
    (TIME_PHRASE 6000-5000/CD_TIME BC/NNP_TIME)
    (TIME_PHRASE tomorrow/NN_TIME)
    (TIME_PHRASE at/IN 2pm/CD_TIME)
    (TIME_PHRASE at/IN 5:00pm/CD_TIME)
    (TIME_PHRASE 1000/CD_TIME)
    
    
    
********************************************************************
2 WAYS OF TESTING THE ALGORITHM
********************************************************************

a). One way to test the algorithm is to get precision and accuracy for the TIME Entity tags by giving in the labelled_data.txt

b). Another way to to test is to visually seeing the output for any given input text in TEST_TEXT and call time_entity_recognizer(TEST_TEXT). We get both 
input = text containing one or many time tags and time phrases
output = gives us (i)time tags and (ii)time phrases for the sentences.

"""



# time pattern strings.
abbre = "(\d?\d:?)?\d?\d\s?(p.?m.?|a.?m.?)|bc|ad|bce|ce"
phrasal_words = "(before|after|earlier|later|ago)"
iso = "\d+[/-]\d+[/-]\d+ \d+:\d+:\d+\.\d+"
year = "((?<=\s)\d{4}|^\d{4})"
tod = "morning|afternoon|evening|night"

#inspiration for some tags from https://code.google.com/p/nltk/source/browse/trunk/nltk_contrib/nltk_contrib/timex.py
numbers = "(one|two|three|four|five|six|seven|eight|nine|ten| \
          eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen| \
          eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty| \
          ninety|hundred|thousand)"
week_day = "(monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
month = "(january|february|march|april|may|june|july|august|september| \
          october|november|december)"
dmy = "(year|day|week|month)"
relation_day = "(today|yesterday|tomorrow|tonight|tonite)"



def my_tokenizer(sentences):    
    sentence_lists = nltk.sent_tokenize(sentences) #default sentence segmenter
    word_tokens = [nltk.word_tokenize(sent) for sent in sentence_lists] #word tokenizer 
    return word_tokens


def pos_tok(sentences):
    """
    ********************************************************************
    1. tokenize the sentences in the text.
    2. apply Parts_of_specch taggers to the text
    ********************************************************************
    """    
    word_tokens = my_tokenizer(sentences)
    #print "\nsentence tokens===",sent_tokens   

    pos_tagged_sents = [nltk.pos_tag(sent) for sent in word_tokens]  
    #print "\npost_tagger==",pos_tagged_sents
    return pos_tagged_sents
    


#TIME PATTERNS
time_patterns=numbers+"|"+week_day+"|"+month+"|"+dmy+"|"+iso+"|"+year+"|"+abbre+"|"+relation_day+"|"+tod+"|"+phrasal_words
re_TIME=re.compile("("+time_patterns+")s*",re.IGNORECASE)
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def append_time(t):
    """
    ********************************************************************
    regex pattern matching on the word tokens
    ********************************************************************
    """
    
    if re_TIME.match(t[0]):
        return (t[0], "MY_TIME")
    else:
        return t


def time_tag(text):
    
    time_tags = []
    sent_time_taglist = []
    for s in pos_tok(text):
        tagged_sent = []
        token_list = []
        
        #tagged_sent = map(lambda x: append_time(x), s)
        #time_tags.append(tagged_sent) 
        
        for token in s:
            if re_TIME.match(token[0]):
                tagged_sent.append((token[0], token[1]+"_TIME"))
                token_list.append(token[0])
            else:
                tagged_sent.append(token)  
        sent_time_taglist.append(token_list)
        time_tags.append(tagged_sent)
         
    return time_tags,sent_time_taglist


    
def time_phrase_chunker(pos_sent):
    """
    ********************************************************************
     Develop aTIME_PHRASE chunker using the regular expression chunker nltk.RegexpParser. 
    ********************************************************************   
    """
    
    time_sym = "N.*TIME|J.*TIME|W.*TIME|C.*TIME|I.*TIME|R.*TIME"    
    grammar = r"""
        TIME_PHRASE: {<NNTIME>*<,|.>?<IN><RB.*>*<DT|CD|"""+time_sym+r""">*<"""+time_sym+r""">+<IN>*<NNTIME>*} # prepositional time phrase
                    {<IN|RB.*>*<IN|RB.*|DT>*<CD|CDTIME|DT|JJ|"""+time_sym+r"""><"""+time_sym+r""">+<RB.*>*} # simple time phrase
                      {<CD><"""+time_sym+r""">}
                      {<"""+time_sym+r""">}  # single time unit
                        NP: {<DT|JJ|NN.*>+}          # Chunk sequences of DT, JJ, NN
                        PP: {<IN><NP>}               # Chunk prepositions followed by NP
                        VP: {<VB.*><NP|PP|CLAUSE>+$} # Chunk verbs and their arguments
                        CLAUSE: {<NP><VP>}           # Chunk NP, VP
                                                                                                         """
    chunkParser = nltk.RegexpParser(grammar, loop=2)
    
    parsed_sents = []
    for s in pos_sent:
        parsed_sents.append(chunkParser.parse(s))
    return parsed_sents



def get_stats(sent_tokens, correct_list, predicted_list):
    """
       ********************************************************************
       Calculating True_pos, True_neg, False_pos and False_neg
       ********************************************************************    
    """      
    
    predicted = set(predicted_list)
    correct = set(correct_list)
    all_tokens = []
    for tokens in sent_tokens:
        all_tokens += tokens
    print '%-100s %25s %25s'  %(all_tokens, correct_list ,predicted_list ) 
    
    all_tokens = set(all_tokens)
    predicted_neg = all_tokens.difference(predicted)
    correct_neg = all_tokens.difference(correct)
    
    true_pos = correct.intersection(predicted)
    true_neg = correct_neg.intersection(predicted_neg)
    
    false_pos = predicted.intersection(correct_neg)
    false_neg = predicted_neg.intersection(correct)
    return len(all_tokens), len(correct), len(predicted), len(true_pos), len(true_neg), len(false_pos), len(false_neg)
    

def test_the_data():
    """
    ********************************************************************
    Testing Precision and Recall of our TIME entity algorithm.
    ********************************************************************    
    """    
    curpath = os.path.abspath(__file__)
    filepath = os.path.dirname(curpath)
    fname = os.path.join(filepath, "labelled_data.txt")

    with open(fname, 'r') as test_data:
        all_tokens_tot =0
        correct_tot =0
        predicted_tot =0
        true_pos_tot=0
        true_neg_tot=0
        false_pos_tot=0
        false_neg_tot=0
        print "*"*150                    
        print '%-100s %25s %25s'  %("SENTENCES", "CORRECT TIME TAGS" ,"PREDICTED TIME TAGS" )
        print '%-100s %25s %25s'  %("*"*50, "*"*15 ,"*"*15 ) 
        
        
        for line in test_data:
            #print line
            text,tag_list = line.split("\t")
            if len(line.split("\t")[1].strip()) ==0:
                correct_list = []
            else:
                correct_list = tag_list.strip().split(",")
            time_tags,sent_time_taglist = time_tag(text)
            combined_tags  = []
            sent_tokens = my_tokenizer(text)
            for tags in sent_time_taglist:
                combined_tags += tags
            
            all_tokens, correct, predicted, true_pos, true_neg, false_pos, false_neg = get_stats(sent_tokens, correct_list, combined_tags)
            all_tokens_tot += all_tokens
            correct_tot +=correct
            predicted_tot += predicted
            true_pos_tot += true_pos
            true_neg_tot += true_neg
            false_pos_tot += false_pos
            false_neg_tot += false_neg            
         
        """
        ********************************************************************
        calculating precision and recall....
        ********************************************************************    
        """         
        if predicted_tot==0:
            if correct_tot==0:
                precision = 1.0
            else:
                precision =0
        else:
            precision = float(true_pos_tot)/predicted_tot
            
        if correct_tot==0:
            if predicted_tot==0:
                recall = 1.0
            else:
                recall = 0.0
        else:
            recall =  float(true_pos_tot)/correct_tot
            
        print "\n********************************************************************\n"            
        print "Num of items tested on ==", all_tokens_tot
        print "Precision ==", precision
        print "Recall==",recall
        print "\n********************************************************************\n"    
        
        

            
def time_entity_recognizer(TEST_TEXT):
    """
    ********************************************************************
     Entity Recognition Algorithm to identify TIME Entity.
    ********************************************************************    
    """
    time_tags,sent_time_taglist = time_tag(TEST_TEXT)
    # print ttag
    pprint(time_tags)
    pprint(sent_time_taglist)
       
    print "\n********************************************************************\n"    
    print " Printing all the chunks..."
    print "\n********************************************************************\n"    
    
    for y,x in enumerate(time_phrase_chunker(time_tags)):
        print 'sentence '+str(y),x
        
    print "\n********************************************************************\n"    
    print " Printing only TIME_PHRASE chunks..."
    print "\n********************************************************************\n"  
    chunks = time_phrase_chunker(time_tags)
    for sent_tree in chunks:
        #print [y.node for y in sent_tree.subtrees()]
        for sub_tree in sent_tree.subtrees():
            if sub_tree.node == "TIME_PHRASE":
                print sub_tree    

if __name__ == '__main__':
    
    #Test data in labelled_data.txt file ---> get  precision and accuracy stats "
    #uncomment below line to evaluate the TIME entity recognition algorithm based on the labelled data
    #test_the_data()    


  
    TEST_TEXT="""
    I went to the store this morning. This is beautiful.
    I went to the building.
    Early last night I went to the building.
    What is now China was inhabited by Homo erectus more than a million years ago. 
    Recent study shows that the stone tools found at Xiaochangliang site are magnetostratigraphically dated to 1.36 million years ago.
    The excavations at Yuanmou and later Lantian show early habitation. 
    Perhaps the most famous specimen of Homo erectus found in China is the so-called Peking Man discovered in 1923-27.
    The Neolithic age in China can be traced back to about 5,0000BC.
    Early evidence for proto-Chinese millet agriculture is radiocarbon-dated to about 7000 BC. 
    Farming gave rise to the Jiahu culture 7000 to 5800BC. 
    At Damaidi in Ningxia, 3,172 cliff carvings dating to 6000-5000 BC have been discovered.
    Let us meet tomorrow at 2pm.
    Shall we go at 5:00pm?
    There are 1000 cookies in the jar."""  
    
    
    #generate TIME tags and TIME_PHRASES for the above blob of text
    time_entity_recognizer(TEST_TEXT)
    
    #how to run this file
    #python entity_recognition_time.py 
    
    