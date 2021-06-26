from os import stat
from re import S
from typing import Sequence
from torchtext import data, datasets

class LanguageModelDataset(data.Dataset):
    def __init__(self, path, fields, max_length=None, **kwargs):
        # Type of fields[0] : tuple or list
        if not isinstance(fields[0], (tuple, list)):
            fields = [('text', fields[0])]

        examples = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if max_length and max_length < len(line.split()):
                    continue
                if line != "":
                    examples.append(data.Example.fromlist(
                        [line], fields))
        
        super(LanguageModelDataset, self).__init__(examples, fields, **kwargs)

class TranslationDataset(data.Dataset):
    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src), len(ex.trg))
    
    def __init__(self, path, exts, fields, max_length=None, **kwargs):
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1])]
        if not path.endswith('.'):
            path += '.'

        src_path, trg_path = tuple(os.path.expanduser(path + x) for x in exts)

        examples = []
        with open(src_path, encoding='utf-8') as src_file, open(trg_path, encoding='utf-8') as trg_file:
            for src_line, trg_line in zip(src_file, trg_file):
                src_line, trg_line = src_line.strip(), trg_line.strip()
                if max_length and max_length < max(len(src_line.split()), len(trg_line.split())):
                    continue
                if src_line != '' and trg_line != '':
                    examples.append(data.Example.fromlist(
                        [src_line, trg_line], fields))
        
        super().__init__(examples, fields, **kwargs)

class TextClsDataLoader(object):
    """
    Data type : (text, label)
    spliter : TAB
    """
    def __init__(self, train_filename, valid_filename,
                 batch_size=64, device=0, max_vocab=999999,
                 min_freq=1, use_eos=False, shuffle=True):

        super(TextClsDataLoader, self).__init__()

        # Define field of the input file.
        # The input file consists of two fields.
        self.label = data.Field(sequential=False, use_vocab=True, unk_token=None)
        self.text = data.Field(use_vocab=True, batch_first=True, include_lengths=False, eos_token='<EOS>' if use_eos else None)
        
        # Those defined two columns will be delimited by TAB.
        # Thus, we use TabularDataset to load two columns in the input file.
        # Files consist of two columns: label field and text field.
        train, valid = data.TabularDataset.splits(path='', train=train_filename, validation=valid_filename,
                                                  format='tsv', field=[('label', self.label), ('text', self.text)])
        
        # Those loaded dataset would be feeded into each iterator: train iterator and valid iterator.
        # We sort input sentences by length, to group similar lengths.
        self.train_iter, self.valid_iter = data.BucketIterator.splits((train, valid), batch_size=batch_size,
                                        device='cuda:%d' % device, shuffle=shuffle, sort_key=lambda x: len(x.text),
                                        sort_within_batch=True)
        
        # At last, we make a vocabulary for label and text field.
        # It is making mapping table between words and indice.
        self.label.build_vocab(train)
        self.text.build_vocab(train, max_size=max_vocab, min_freq=min_freq)
    



class SingleCorpusDataLoader(object):
    def __init__(self, train_filename, valid_filename, batch_size=64, device=0,
                 max_vocab=9999999, max_length=255, fix_length=None, use_bos=True,
                 use_eos=True, shuffle=True):

        super(SingleCorpusDataLoader, self).__init__()
        
        self.text = data.Field(sequential=True, use_vocab=True, include_lengths=True,
                               fix_length=fix_length, init_token='<BOS>' if use_bos else None,
                               eos_token='<EOS>' if use_eos else None)
        
        train = LanguageModelDataset(path=train_filename, fields=[('text', self.text)], max_length=max_length)  
        valid = LanguageModelDataset(path=valid_filename, fields=[('text', self.text)], max_length=max_length)

        self.train_iter = data.BucketIterator(train, batch_size=batch_size, device='cuda:%d' % device, shuffle=shuffle,
                                              sort_key=lambda x: -len(x.text), sort_within_batch=True)
        self.valid_iter = data.BucketIterator(valid, batch_size=batch_size, device='cuda:%d' % device, shuffle=False,
                                              sort_key=lambda x: -len(x.text), sort_within_batch=True)
        self.text.build_vocab(train, max_size=max_vocab)


class ParallelCorpusDataLoader(object):
    def __init__(self, train_filename=None, valid_filename=None, exts=None, batch_size=64,
                 device=0, max_vocab=99999999, max_length=255, fix_length=None, use_bos=True,
                 use_eos=True, shuffle=True, dsl=False):
        super(ParallelCorpusDataLoader, self).__init__()

        self.src = data.Field(sequential=True, use_vocab=True, batch_first=True, include_lengths=True,
                              fix_length=fix_length, init_token='<BOS>' if dsl else None,
                              eos_token='<EOS>' if dsl else None)

        self.tgt = data.Field(sequential=True, use_vocab=True, batch_first=True, include_lengths=True,
                              fix_length=fix_length, init_token='<BOS>' if use_bos else None,
                              eos_token='<EOS>' if use_eos else None)
        
        if train_filename is not None and valid_filename is not None and exts is not None:
            train = TranslationDataset(path=train_filename, exts=exts,
                                       fields=[('src', self.src), ('tgt', self.tgt)],
                                       max_length=max_length)
            valid = TranslationDataset(path=valid_filename, exts=exts,
                                       fields=[('src', self.src), ('tgt', self.tgt)],
                                       max_length=max_length)
            
            self.train_iter = data.BucketIterator(train, batch_size=batch_size, device='cuda:%d' % device,
                                                  shuffle=shuffle, sort_key=lambda x: len(x.tgt) + (max_length * len(x.src)),
                                                  sort_within_batch=True)
            self.valid_iter = data.BucketIterator(valid, batch_size=batch_size, device='cuda:%d' % device,
                                                  shuffle=False, sort_key=lambda x: len(x.tgt) + (max_length * len(x.src)),
                                                  sort_within_batch=True)
            
            self.src.build_vocab(train, max_size=max_vocab)
            self.tgt.build_vocab(train, max_size=max_vocab)
    
    def load_vocab(self, src_vocab, tgt_vocab):
        self.src.vocab = src_vocab
        self.tgt.vocab = tgt_vocab