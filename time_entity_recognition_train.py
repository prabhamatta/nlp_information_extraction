# Natural Language Toolkit: code_unigram_chunker
"""
References:
http://streamhacker.wordpress.com/2008/12/29/how-to-train-a-nltk-chunker/
http://freshlyminted.co.uk/blog/2011/02/28/getting-band-and-artist-names-nltk/
https://code.google.com/p/nltk/source/browse/trunk/nltk_contrib/nltk_contrib/timex.py
http://nltk.org/book/ch07.html

Notes:
    This file requires that train.txt be in the same directory
    The runtime for this is very slow: ~162 seconds
    algorithm='iis' is not the best algorithm, but it works without any extra packages. A bet algorithm might yield better results
    
    ChunkParse score:
        IOB Accuracy:  63.9%
        Precision:     27.3%
        Recall:        30.0%
        F-Measure:     28.6%

Sample sentence parsing:

[('30', 'CD'), ('years', 'NNS'), ('before', 'IN'), ('two', 'CD'), ('hours', 'NNS'), ('ago', 'RB'), ('.', '.')]
Parsed:
(S
  (TEMPORAL 30/CD years/NNS)
  before/IN
  (TEMPORAL two/CD hours/NNS ago/RB)
  ./.)

[('It', 'PRP'), ('happened', 'VBD'), ('on', 'IN'), ('February', 'NNP'), ('29', 'CD'), (',', ','), ('2012', 'CD'), ('.', '.')]
Parsed:
(S
  It/PRP
  happened/VBD
  on/IN
  (DATE February/NNP 29/CD ,/, 2012/CD)
  ./.)

[('Let', 'NNP'), ('us', 'PRP'), ('meet', 'VBP'), ('tomorrow', 'NN'), ('at', 'IN'), ('2pm', 'CD'), ('.', '.')]
Parsed:
(S Let/NNP us/PRP meet/VBP (DATE tomorrow/NN at/IN 2pm/CD) ./.)

[('Shall', 'NNP'), ('we', 'PRP'), ('go', 'VBP'), ('at', 'IN'), ('5:00pm', 'CD'), ('?', '.')]
Parsed:
(S Shall/NNP we/PRP go/VBP at/IN (DATE 5:00pm/CD) ?/.)

[('I', 'PRP'), ('went', 'VBD'), ('to', 'TO'), ('the', 'DT'), ('store', 'NN'), ('this', 'DT'), ('morning', 'NN'), ('.', '.')]
Parsed:
(S
  I/PRP
  went/VBD
  to/TO
  the/DT
  store/NN
  (DATE this/DT morning/NN)
  ./.)

[('September', 'NNP'), ('14', 'CD'), (',', ','), ('1939', 'CD'), ('-', ':'), ('October', 'NNP'), ('8', 'CD'), (',', ','), ('1946', 'CD'), ('.', '.')]
Parsed:
(S
  (DATE
    September/NNP
    14/CD
    ,/,
    1939/CD
    -/:
    October/NNP
    8/CD
    ,/,
    1946/CD)
  ./.)

[('December', 'NNP'), ('12', 'CD'), (',', ','), ('1822', 'CD'), ('to', 'TO'), ('March', 'NNP'), ('1', 'CD'), (',', ','), ('1977', 'CD'), ('.', '.')]
Parsed:
(S
  (DATE
    December/NNP
    12/CD
    ,/,
    1822/CD
    to/TO
    March/NNP
    1/CD
    ,/,
    1977/CD)
  ./.)


[('The', 'DT'), ('Bolshevik', 'NNP'), ('revolution', 'NN'), ('started', 'VBD'), ('on', 'IN'), ('October', 'NNP'), ('25', 'CD'), (',', ','), ('1917', 'CD'), ('.', '.')]
Parsed:
(S
  The/DT
  Bolshevik/NNP
  revolution/NN
  started/VBD
  (DATE on/IN October/NNP 25/CD ,/, 1917/CD)
  ./.)

[Finished in 162.9s]
"""
import nltk
from nltk.chunk.util import conlltags2tree
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

SMALL_PARSE_SET="""30 years before two hours ago.
It happened on February 29, 2012.
"""

CHUNK_TYPES=('DATE', 'TEMPORAL', 'PER', 'LOC', 'GPE')


TEST_TEXT="""Let us meet tomorrow at 2pm.
Shall we go at 1:20pm?
I went to the store this afternoon.
September 14, 1939 - October 8, 1946.
It went from December 12, 1822 to March 1, 1977.
The Bolshevik revolution started on October 25, 1917.
I get calls almost every day at 11am.
30 years before two hours ago.
It happened on February 29, 2012.
"""

def getTrain1(path='train.txt'):
    t = file(path).read()
    ret = t.split('\n\n')
    # for x in t.split('\n\n'):
    #     # r = x.split('\n')
    #     ret.append(x)
    return ret

def conllFmt(sents):
    ret = []
    ret = map(lambda x: nltk.chunk.conllstr2tree(x,CHUNK_TYPES), sents)
    return ret


def to_chunker_training_fmt(sent):
    train_data = [[(t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)]
                      for sent in train_sents]
    return train_data

class UnigramChunker(nltk.ChunkParserI):
    def __init__(self, train_sents): # [_code-unigram-chunker-constructor]
        train_data = [[(t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)]
                      for sent in train_sents]
        self.tagger = nltk.UnigramTagger(train_data) # [_code-unigram-chunker-buildit]

    def parse(self, sentence): # [_code-unigram-chunker-parse]
        pos_tags = [pos for (word,pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [(word, pos, chunktag) for ((word,pos),chunktag)
                     in zip(sentence, chunktags)]
        return conlltags2tree(conlltags)


#################################
class ConsecutiveNPChunkTagger(nltk.TaggerI): # [_consec-chunk-tagger]

    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = npchunk_features(untagged_sent, i, history) # [_consec-use-fe]
                train_set.append( (featureset, tag) )
                history.append(tag)
        self.classifier = nltk.MaxentClassifier.train( # [_consec-use-maxent]
            train_set, algorithm='iis', trace=0)

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = npchunk_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)

class ConsecutiveNPChunker(nltk.ChunkParserI): # [_consec-chunker]
    def __init__(self, train_sents):
        tagged_sents = [[((w,t),c) for (w,t,c) in
                         nltk.chunk.tree2conlltags(sent)]
                        for sent in train_sents]
        self.tagger = ConsecutiveNPChunkTagger(tagged_sents)

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w,t,c) for ((w,t),c) in tagged_sents]
        return conlltags2tree(conlltags)

############################################
def npchunk_features(sentence, i, history):
    word, pos = sentence[i]
    if i == 0:
        prevword, prevpos = "<START>", "<START>"
    else:
        prevword, prevpos = sentence[i-1]
    if i == len(sentence)-1:
        nextword, nextpos = "<END>", "<END>"
    else:
        nextword, nextpos = sentence[i+1]
    return {"pos": pos, 
            "word": word, 
            "prevpos": prevpos,
            "nextpos": nextpos, # [_chunk-fe-lookahead]
            "prevpos+pos": "%s+%s" % (prevpos, pos),  # [_chunk-fe-paired]
            "pos+nextpos": "%s+%s" % (pos, nextpos),
            "tags-since-dt": tags_since_dt(sentence, i)}  # [_chunk-fe-complex]

def tags_since_dt(sentence, i):
    tags = set()
    for word, pos in sentence[:i]:
        if pos == 'DT':
            tags = set()
        else:
            tags.add(pos)
    return '+'.join(sorted(tags))
#####################################

def pos_tok(text):
    ss = []
    for s in tokenizer.tokenize(text):
        # tmp =nltk.pos_tag(nltk.word_tokenize(s))
        # tmp = filter(lambda x: x[0] not in '.,?/\\-=+' ,s)
        ss.append(nltk.pos_tag(nltk.word_tokenize(s)))
    return ss


def simple_format(text):
    ret =''
    for ss in pos_tok(text):
        for t in ss:
            ret+=t[0]+' '+t[1]+' '+'O\n'
        ret+='\n'
    return ret


def chunk_sents(chunker,sents):
    for x in sents:
        print x
        print 'Parsed:'
        print chunker.parse(x)
        print '\n'

def main():
    sents = getTrain1()
    trees = conllFmt(sents)
    
    tst = trees[0:8]
    trn = trees[8:]
    
    chunker = ConsecutiveNPChunker(trn)
    print chunker.evaluate(tst)
    chunker = ConsecutiveNPChunker(trees)
    chunk_sents(chunker, pos_tok(TEST_TEXT))

if __name__ == '__main__':
    main()

    