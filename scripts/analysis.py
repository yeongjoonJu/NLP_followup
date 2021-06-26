import pandas as pd
import numpy as np
from utils import KoreanPreprocessor

def get_term_frequency(doc, word_dict=None):
    assert type(doc)==list
    if word_dict is None:
        word_dict = {}

    for w in doc:
        if word_dict.get(w) is None:
            word_dict[w] = 1
        else:
            word_dict[w] += 1
    
    return word_dict

def get_document_frequency(docs):
    tfs = []
    vocab = set([])
    df = {}
    
    for doc in docs:
        tf = get_term_frequency(doc)
        tfs += [tf]
        # add words
        vocab = vocab | set(tf.keys())
    
    for v in list(vocab):
        df[v] = 0
        for tf in tfs:
            if tf.get(v) is not None:
                df[v] += 1
    
    return df

def get_doc_freq_from_tfs(tfs, vocab):
    df = {}
    for v in list(vocab):
        df[v] = 0
        for tf in tfs:
            if tf.get(v) is not None:
                df[v] += 1
    return df

def get_tfidf(docs, titles=None, n_limit=None):
    if n_limit is not None and type(n_limit) is int:
        docs = docs[:n_limit]
        titles = titles[:n_limit]

    # Calculate TF-IDF
    vocab = {}
    tfs = []
    for d in docs:
        tf = get_term_frequency(d)
        for w, freq in tf.items():
            if vocab.get(w) is None:
                vocab[w] = freq
            else:
                vocab[w] += freq
        tfs += [tf]
    df = get_doc_freq_from_tfs(tfs, vocab)

    print('The number of words', len(vocab.keys()))

    stats = []
    if titles is None:
        titles = ['doc'+str(i+1) for i in range(len(docs))]

    w = 1
    for word, freq in vocab.items():
        print('\r%d %s' % (w, word), end='')
        tfidfs = []
        for idx in range(len(docs)):
            if tfs[idx].get(word) is not None:
                tfidfs.append(tfs[idx][word] * np.log(len(docs) / df[word]))
            else:
                tfidfs += [0]
        
        stats.append((word, freq, *tfidfs, max(tfidfs)))
        w += 1
    print()
    
    return pd.DataFrame(stats, columns=('word', 'freq', *titles, 'max')).sort_values('max', ascending=False)

if __name__=='__main__':
    processor = KoreanPreprocessor()
    dt = processor.get_segmented_docs(js_filename='nate_pann_ranking20210123_20210622_rm_noisy.json')
    get_tfidf(dt['docs'], dt['titles'], 50).to_csv('tf_idf.csv', index=False, encoding="utf-8-sig")