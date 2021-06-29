import urllib.request
from konlpy import tag
import pandas as pd
from konlpy.tag import Mecab
# from soynlp.word import WordExtractor
import json, re
from nltk import FreqDist, tokenize
import numpy as np

class KoreanPreprocessor(object):
    def __init__(self, doc_key='content', stopwords=None):
        self.mecab = Mecab()
        self.doc_key = doc_key
        self.single_stopwords = []
        self.phrase_stopwords = ""
        if stopwords is None:
            with open('data/korean_stopwords.txt', 'rt') as f:
                stopwords = f.readlines()
            
            for word in stopwords:
                word = word.strip()
                if len(word.split(' ')) == 1:
                    self.single_stopwords.append(word)
                else:
                    self.phrase_stopwords += "("+word+")|"
            
            if self.phrase_stopwords!="":
                self.phrase_stopwords = self.phrase_stopwords[:-1]
        else:
            self.single_stopwords = stopwords

    def read_js(self, js_filename):
        with open(js_filename, 'r') as js:
            docs = json.load(js)
        return docs
        
    def remove_noises(self, doc, strict=False, stopword=False):
        # print(doc)
        # Remove video tag
        doc = re.sub(r'(addLoadEvent)[^\n]+;', ' ', doc)

        # Remove line alignments
        doc = re.sub(r'([\n\s])\1{2,}', r'\1', doc)

        # Remove links
        doc = re.sub(r'((http)([s:/]{3,4})?)?[a-z0-9]+([\.][a-z0-9]+)+[\S]+', ' ', doc)

        # Remove special char
        doc = re.sub(r'[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9\s\?\!\.]', ' ', doc)
        
        if strict:
            # Remove chosung, punct, and digits
            doc = re.sub(r'[ㄱ-ㅎㅏ-ㅣ?!\.]+', ' ', doc)
            doc = re.sub(r'[0-9]+', 'nu', doc)
        else:
            # Reduce repetition
            doc = re.sub(r'([ㄱ-ㅎㅏ-ㅣ\.\s])\1{2,}', r'\1\1', doc)

        # Reduce repetition ? ! .
        doc = re.sub(r'(.)\1{3,}', r'\1\1\1', doc)
        doc = re.sub(r'([?!]){2,}', r'\1', doc)
        
        if stopword:
            doc = self.remove_stopwords(doc)
        doc = re.sub(r'[\s]{2,}', ' ', doc)

        return doc.strip()

    def remove_stopwords(self, doc):
        new_doc = []
        doc = doc.split(' ')
        for word in doc:
            if not word.strip() in self.single_stopwords:
                new_doc.append(word)
        new_doc = " ".join(new_doc)

        if self.phrase_stopwords != "":
            new_doc = re.sub(r"%s" % self.phrase_stopwords, "", new_doc)

        return new_doc

    def tokenize(self, doc, flatten=True, tagging=False):
        # doc = self.mecab.morphs(doc)
        if flatten:
            doc = self.mecab.pos(doc)
            tokenized_docs = []
            for word in doc:
                # 조사와 어미 제거
                if word[1][0] in ['E', 'J']:
                    continue
                if tagging:
                    tokenized_docs.append(word)
                else:
                    tokenized_docs.append(word[0])
        else:
            tokenized_docs = []
            doc = doc.split('\n')
            for sent in doc:
                sent = self.mecab.pos(sent)
                tokenized = []
                for word in sent:
                    # 조사와 어미 제거
                    if word[1][0] in ['E', 'J']:
                        continue
                    if tagging:
                        tokenized.append(word)
                    else:
                        tokenized.append(word[0])

                if len(tokenized)!=0:
                    tokenized_docs.append(tokenized)
        
        return tokenized_docs

    def remove_empty_docs_and_noises(self, docs=None, js_filename=None, strict=True, to_json=True):
        if js_filename is not None:
            docs = self.read_js(js_filename)
        
        renew_docs = []
        for idx in range(len(docs)):
            doc = self.remove_noises(docs[idx][self.doc_key], strict=strict)
            if len(doc) != 0:
                docs[idx]['title'] = self.remove_noises(docs[idx]['title'])
                docs[idx][self.doc_key] = doc
                renew_docs.append(docs[idx])

        print('The number of removed documents', len(docs) - len(renew_docs))
        if to_json:
            with open(js_filename[:-5]+'_rm_noisy.json', 'w') as js:
                json.dump(renew_docs, js, ensure_ascii=False)

        return renew_docs

    def get_segmented_docs(self, docs=None, js_filename=None, rm_noise=False, add_title=False):
        if js_filename is not None:
            docs = self.read_js(js_filename)
        
        titles = []
        seg_docs = []

        if type(docs[0]) is dict:
            for doc in docs:
                if add_title and doc.get('title') is not None:
                    titles.append(doc['title'])
                if rm_noise:
                    doc[self.doc_key] = self.remove_noises(doc[self.doc_key])
                seg_docs.append([self.tokenize(doc[self.doc_key], flatten=False)])

            if len(titles)==len(seg_docs):
                seg_docs = {'titles':titles, 'docs': seg_docs}

        elif type(docs[0]) is list:
            for doc in docs:
                if rm_noise:
                    doc = self.remove_noises(doc)
                seg_docs.append([self.tokenize(doc, flatten=False)])
        else:
            raise TypeError

        return seg_docs
        

def get_vocab(docs, processor=None, rm_stopwords=False, vocab_size=None, tagging=False):
    # docs : list ["doc", "doc", ...]
    if processor is None:
        processor = KoreanPreprocessor()

    vocab = []
    for doc in docs:
        doc = processor.remove_noises(doc, strict=True, stopword=rm_stopwords)
        doc = processor.tokenize(doc, flatten=True, tagging=tagging)
        
        if tagging:
            doc = [word[0]+'_'+word[1] for word in doc]
        vocab.append(doc)
    
    vocab = FreqDist(np.hstack(vocab))
    print('Found words', len(vocab))
    if vocab_size is not None:
        vocab = vocab.most_common(vocab_size)
    else:
        _vocab = []
        for word, freq in vocab.items():
            if freq >= 5:
                _vocab.append((word, freq))
        vocab = _vocab
    
    # vocab_idx = {}
    # for idx, word in enumerate(vocab):
    #     vocab_idx[word[0]] = idx + 2
    
    # vocab_idx['pad'] = 1
    # vocab_idx['unk'] = 0
    # return vocab_idx

    print('Constructed vocab', len(vocab))
    
    vocab_freq = {}
    for word, freq in vocab:
        vocab_freq[word] = freq

    return vocab_freq            



def download_naver_sentiment():
    urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt")


if __name__=='__main__':
    # Make vocabulary based Nate pann ranking documents
    processor = KoreanPreprocessor()
    data = processor.read_js('data/nate_pann_ranking20200901_20210628.json')
    docs = []
    for sample in data:
        docs.append(sample['content'])

    vocab = get_vocab(docs, vocab_size=None, rm_stopwords=True, tagging=False)
    with open('data/vocab_freq.js', 'w') as js:
      json.dump(vocab, js, ensure_ascii=False)

    # processor = KoreanPreprocessor()
    # processor.remove_empty_docs_and_noises(js_filename='data/nate_pann_ranking20200901_20210628.json')

    # Make vocabs
    # processor = KoreanPreprocessor()
    # data = processor.read_js('data/nate_pann_ranking20210123_20210622.json')
    # docs = []
    # for sample in data:
    #     docs.append(sample['content'])

    # for doc in docs:
    #     doc = processor.remove_noises(doc, strict=True)
    #     doc = processor.tokenize(doc, flatten=False)

        