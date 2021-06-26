import urllib.request
import pandas as pd
from konlpy.tag import Mecab
# from soynlp.word import WordExtractor
import json, re
from nltk import FreqDist
import numpy as np

class KoreanPreprocessor(object):
    def __init__(self, doc_key='content', stopwords=None):
        self.mecab = Mecab()
        self.doc_key = doc_key
        if stopwords is None:
            with open('data/korean_stopwords.txt', 'rt') as f:
                stopwords = f.readlines()
            stopwords = [word.strip() for word in stopwords]
        self.stopwords = stopwords


    def read_docs_from_js(self, js_filename):
        with open(js_filename, 'r') as js:
            docs = json.load(js)
        return docs
        
    def remove_noises(self, doc, strict=False, stopword=False):
        # print(doc)
        # Remove video tag
        doc = re.sub(r'(addLoadEvent)[^\n]+;', ' ', doc)

        # Remove line alignments
        doc = re.sub(r'[\n\s]+', ' ', doc)

        # Remove links
        doc = re.sub(r'((http)([s:/]{3,4})?)?[a-z0-9]+([\.][a-z0-9]+)+[\S]+', ' ', doc)

        # Remove special char
        doc = re.sub(r'[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9\s\?\!\.]', ' ', doc)
        
        if strict:
            # Remove chosung, punct, and digits
            doc = re.sub(r'[ㄱ-ㅎㅏ-ㅣ\s\.]+', ' ', doc)
            doc = re.sub(r'[0-9]+', '_digit', doc)
        else:
            # Reduce repetition
            doc = re.sub(r'([ㄱ-ㅎㅏ-ㅣ\.\s])\1{2,}', r'\1\1', doc)

        # Reduce repetition ? ! .
        doc = re.sub(r'([?!]){2,}', r'\1', doc)
        doc = re.sub(r'[\.]{3,}', '..', doc)
        doc = re.sub(r'[\s]{2,}', ' ', doc)

        return doc.strip()

    def tokenize(self, doc):
        doc = self.mecab.morphs(doc)
        return doc

    def remove_stopwords(self, tokens):
        words = []
        for word in tokens:
            if not word in self.stopwords:
                words.append(word)

        return words

    def remove_empty_docs_and_noises(self, docs=None, js_filename=None, strict=True, to_json=True):
        if js_filename is not None:
            docs = self.read_docs_from_js(js_filename)
        
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

    def get_segmented_docs(self, docs=None, js_filename=None, rm_noise=False):
        if js_filename is not None:
            docs = self.read_docs_from_js(js_filename)
        
        titles = []
        seg_docs = []
        for doc in docs:
            if doc.get('title') is not None:
                titles.append(doc['title'])
            if rm_noise:
                doc[self.doc_key] = self.remove_noises(doc[self.doc_key])
            seg_docs.append(self.tokenize(doc[self.doc_key]))

        if len(titles)==len(seg_docs):
            return {'titles':titles, 'docs': seg_docs}
        else:
            return seg_docs


def get_vocab(docs, processor=None, rm_stopwords=False, vocab_size=None):
    # docs : list ["doc", "doc", ...]
    if processor is None:
        processor = KoreanPreprocessor()
    
    vocab = []
    for doc in docs:
        doc = processor.remove_noises(doc, strict=True)
        doc = processor.tokenize(doc)
        if rm_stopwords:
            doc = processor.remove_stopwords(doc)
        vocab.append(doc)
    
    vocab = FreqDist(np.hstack(vocab))
    print('Found words', len(vocab))
    if vocab_size is not None:
        vocab = vocab.most_common(vocab_size)
    
    vocab_idx = {}
    for idx, word in enumerate(vocab):
        vocab_idx[word[0]] = idx + 2
    
    vocab_idx['pad'] = 1
    vocab_idx['unk'] = 0

    return vocab_idx


def download_naver_sentiment():
    urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt")


if __name__=='__main__':
    # Make vocabulary based Nate pann ranking documents
    processor = KoreanPreprocessor()
    data = processor.read_docs_from_js('data/nate_pann_ranking20210123_20210622.json')
    docs = []
    for sample in data:
        docs.append(sample['content'])

    vocab = get_vocab(docs, vocab_size=5000, rm_stopwords=True)
    with open('data/vocab.js', 'w') as js:
        json.dump(vocab, js, ensure_ascii=False)

    # processor = KoreanPreprocessor()
    # processor.remove_empty_docs_and_noises(js_filename='data/nate_pann_ranking20210123_20210622.json')