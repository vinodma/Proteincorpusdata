%load_ext autoreload
%autoreload 2
%matplotlib inline

from snorkel import SnorkelSession
session = SnorkelSession()
from snorkel.parser import TSVDocParser
from snorkel.parser import TSVDocParser
doc_parser = TSVDocParser(path='data/proteincorpus_sm.tsv')
from snorkel.parser import SentenceParser

sent_parser = SentenceParser()

from snorkel.parser import CorpusParser

cp = CorpusParser(doc_parser, sent_parser)
%time corpus = cp.parse_corpus(session, 'Protein Training')

for name, path in [('Protein Development', 'data/protein_dev.tsv'),
                   ('Protein Test', 'data/protein_test.tsv')]:
    doc_parser.path=path
    %time corpus = cp.parse_corpus(session, name)
    session.commit()


from snorkel import SnorkelSession
session = SnorkelSession()
from snorkel.models import Corpus

corpus = session.query(Corpus).filter(Corpus.name == 'Protein Training').one()
corpus

sentences = set()
for document in corpus:
    for sentence in document.sentences:
        sentences.add(sentence)

from snorkel.candidates import Ngrams
from snorkel.models import candidate_subclass
#entity = candidate_subclass('entity', ['entity1', 'entity2'])
import pandas as pd
ROOT = 'data/dicts/'
proteins   = set(pd.read_csv(ROOT + 'protein_names.csv', header=None, index_col=0, encoding='utf-8').dropna()[1])
ngrams = Ngrams(n_max=1)
from snorkel.matchers import DictionaryMatch

longest_match_only = True
dict_proteins = DictionaryMatch(d=proteins, ignore_case=True, 
                                longest_match_only=longest_match_only)
#misc_matcher = MiscMatcher(longest_match_only=True)
from snorkel.candidates import CandidateExtractor
ce = CandidateExtractor(entity, [ngrams, ngrams], [dict_proteins, dict_proteins],
                        symmetric_relations=False, nested_relations=False, self_relations=False)

%time c = ce.extract(sentences, 'Protein1 Training Candidates', session)



for corpus_name in ['Protein Development']:
    corpus = session.query(Corpus).filter(Corpus.name == corpus_name).one()
    sentences = set()
    for document in corpus:
        for sentence in document.sentences:
            sentences.add(sentence)
    
    %time c = ce.extract(sentences, 'Protein1 Development Candidates', session)
    session.add(c)
session.commit()


%load_ext autoreload
%autoreload 2

from snorkel import SnorkelSession
session = SnorkelSession()
from snorkel.models import CandidateSet
from snorkel.models import candidate_subclass
#entity = candidate_subclass('entity', ['entity1', 'entity2'])

train = session.query(CandidateSet).filter(CandidateSet.name == 'Protein1 Training Candidates').one()




from snorkel.annotations import FeatureManager

feature_manager = FeatureManager()

%time F_train = feature_manager.create(session, train, 'Train1 Features')


import re
from snorkel.lf_helpers import get_left_tokens, get_right_tokens, get_between_tokens, get_text_between



rightbeforeorafter= {'interaction','interaction between'}
betweentags={'and'}
negativetags={'as','was','impact','its','nm','set'}

def LF_too_far_apart(c):
    return -1 if len(get_between_tokens(c)) > 10 else 0


    
def LF_nospace(c):
    return 1 if [] == get_between_tokens(c)  else 0

#def LF_betweentags(c):
#    return 1 if len(betweentags.intersection(set(get_between_tokens(c)))) >0  and len(get_between_tokens(c)) < 5 else 0

def LF_right_before_or_after(c):
    if len(rightbeforeorafter.intersection(set(c[0].parent.words))) == 0:
        if len(rightbeforeorafter.intersection(set(get_left_tokens(c[0],window=5, attrib='words')))) > 0 or len(rightbeforeorafter.intersection(set(get_right_tokens(c[0],window=5, attrib='words')))) > 0:
            return 1
        else:
            return 0
    else:
        return 0

def LF_betweentokens(c):
    if len(get_between_tokens(c)) < 2:
        return 1
    else:
        return 0
    

LFs = [LF_too_far_apart,LF_nospace,LF_right_before_or_after,LF_betweentokens]



from snorkel.annotations import LabelManager

label_manager = LabelManager()
%time L_train = label_manager.create(session, train, 'LF2 Labels', f=LFs)

from snorkel.learning import NaiveBayes

gen_model = NaiveBayes()
gen_model.train(L_train, n_iter=1000, rate=1e-5)

from snorkel.learning import NaiveBayes

gen_model = NaiveBayes()
gen_model.train(L_train, n_iter=1000, rate=1e-5)
train_marginals = gen_model.marginals(L_train)

from snorkel.learning import LogReg
from snorkel.learning_utils import RandomSearch, ListParameter, RangeParameter

iter_param = ListParameter('n_iter', [250, 500, 1000, 2000])
rate_param = RangeParameter('rate', 1e-4, 1e-2, step=0.75, log_base=10)
reg_param  = RangeParameter('mu', 1e-8, 1e-2, step=1, log_base=10)

disc_model = LogReg()

%load_ext autoreload
%autoreload 2
%matplotlib inline

from snorkel import SnorkelSession
session = SnorkelSession()
from snorkel.models import CandidateSet
from snorkel.models import candidate_subclass
from snorkel.annotations import FeatureManager

feature_manager = FeatureManager()
entity = candidate_subclass('entity', ['entity1', 'entity2'])
dev = session.query(CandidateSet).filter(CandidateSet.name == 'Protein1 Development Candidates').one()
%time F_dev = feature_manager.update(session, dev, 'Train1 Features', False)

from snorkel.annotations import LabelManager

label_manager = LabelManager()
L_gold_dev = label_manager.load(session, dev, "Sotera User")
gold_dev_set = session.query(CandidateSet).filter(CandidateSet.name == 'Protein Development Candidates').one()


from snorkel.learning import LogReg
from snorkel.learning_utils import RandomSearch, ListParameter, RangeParameter

iter_param = ListParameter('n_iter', [250, 500, 1000, 2000])
rate_param = RangeParameter('rate', 1e-4, 1e-2, step=0.75, log_base=10)
reg_param  = RangeParameter('mu', 1e-8, 1e-2, step=1, log_base=10)

disc_model = LogReg()
from snorkel.models import CandidateSet
from snorkel.models import candidate_subclass
#entity = candidate_subclass('entity', ['entity1', 'entity2'])

train = session.query(CandidateSet).filter(CandidateSet.name == 'Protein1 Training Candidates').one()
%time F_train = feature_manager.load(session, train, 'Train1 Features')
from snorkel.annotations import LabelManager

label_manager = LabelManager()
  
%time L_train = label_manager.load(session, train, 'LF2 Labels')
from snorkel.learning import NaiveBayes

gen_model = NaiveBayes()
gen_model.train(L_train, n_iter=1000, rate=1e-5)
train_marginals = gen_model.marginals(L_train)
searcher = RandomSearch(disc_model, F_train, train_marginals, 10, iter_param, rate_param, reg_param)


x = disc_model.predict(F_train)
for iter,index in enumerate(x):
    if(index==1):
        ent1=train[iter][0].get_span()
        ent2=train[iter][1].get_span()
        if(ent1 != ent2):
            print ent1 +"," + ent2
            