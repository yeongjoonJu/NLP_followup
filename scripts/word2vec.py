from re import sub
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
from utils import *
import numpy as np
import math, os, pickle, random, gc

def get_unigram_prob(vocab_freq):
    num_words = 0
    for key in vocab_freq.keys():
        num_words += vocab_freq[key]

    vocab_prob = {}
    for word, freq in vocab_freq.items():
        vocab_prob[word] = freq / num_words
    
    return vocab_prob

def sampling_from_unigram(vocab_prob, vocab_idx):    
    """sampling from noise distribution (unigram^(3/4)/Z)
    -> After: sampling with uniform distribution
    Args:
        vocab_prob : vocabulary with probability from unigram distribution
    Returns:
        list sampled with unigram distribution
    """
    samples = []
    for key in vocab_prob.keys():
        # (unigram_dist)^(3/4)/Z
        samples.extend([vocab_idx[key]] * int(math.pow(vocab_prob[key], 3/4)/0.001))

    return samples

def get_subsampling_probs(vocab_prob, thres=1e-5):
    """get probabilities for subsampling
    Args:
        vocab_prob : vocabulary with probabilities from unigram distribution
        thres : chosen theshold t (1e-5 in the paper)
    Returns:
        dict (key: word, value: discarding probility)
    """
    sub_prob = {}
    for word, prob in vocab_prob.items():
        sub_prob[word] = 1. - math.pow(thres / prob, 0.5)
        if sub_prob[word] <= 0:
            sub_prob[word] = 0
    
    return sub_prob

def build_skip_gram_dataset(docs_filename, vocab_filename, window_size=2, processor=None):
    """
    Args:
        docs_filename : json filename containing documents
        vocab_filename : json filename containing vocabulary with frequency
    Returns:
        list of [center word idx, context word idx]
        list for noise sampling
        dict for vocabulary (key: word, value: idx)
        dict for context words (key: word, value: list of indices of context words)
        list of probabilities for subsampling
    """
    if processor is None:
        processor = KoreanPreprocessor()
    
    vocab_freq = processor.read_js(vocab_filename)
    docs = processor.read_js(docs_filename)
    docs = [doc['content'] for doc in docs]

    for d in range(len(docs)):
        docs[d] = processor.tokenize(docs[d], flatten=False)

    vocab_prob = get_unigram_prob(vocab_freq)
    subsampling_prob = get_subsampling_probs(vocab_prob, thres=1e-5)

    # Generate voca to idx
    vocab_idx = {}
    for idx, word in enumerate(vocab_freq.keys()):
        vocab_idx[word] = idx + 2
    vocab_idx['pad'] = 1
    vocab_idx['unk'] = 0

    # Generate index pairs with subsampling
    ctx_indices = {}
    idx_pairs = []
    sampling_probs = []
    for d, doc in enumerate(docs):
        print('\rdoc %d/%d' % (d+1, len(docs)), end='')
        for sent in doc:
            sent_len = len(sent)
            for p in range(sent_len):
                if vocab_idx.get(sent[p]) is None:
                    continue
                ctx_indices[vocab_idx[sent[p]]] = []
                for w in range(-window_size, window_size+1):
                    if w==0 or p+w < 0 or p+w >= sent_len:
                        continue
                    if vocab_idx.get(sent[p+w]) is not None:
                        idx_pairs.append([vocab_idx[sent[p]], vocab_idx[sent[p+w]]])
                        sampling_probs.append(subsampling_prob[sent[p]])
                        ctx_indices[vocab_idx[sent[p]]].append(vocab_idx[sent[p+w]])
    print()

    samples_for_noise = sampling_from_unigram(vocab_prob, vocab_idx)

    return idx_pairs, samples_for_noise, vocab_idx, ctx_indices, sampling_probs
    

class SkipGramDataset(Dataset):
    """
    Implementation of https://proceedings.neurips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf
    """
    def __init__(self, docs_filename, vocab_filename, window_size=2, processor=None, neg_samples=5):
        if processor is None:
            processor = KoreanPreprocessor()

        vocab = processor.read_js(vocab_filename)
        self.vocab_size = len(vocab.keys()) + 2
        
        if os.path.exists('data/idx_pairs.list') and os.path.exists('data/noise_samples.list') \
            and os.path.exists('data/vocab_idx.dict') and os.path.exists('data/ctx_indices.dict'):
            idx_pairs = pickle.load(open('data/idx_pairs.list', 'rb'))
            noise_dist = pickle.load(open('data/noise_samples.list', 'rb'))
            vocab_idx = pickle.load(open('data/vocab_idx.dict', 'rb'))
            ctx_indices = pickle.load(open('data/ctx_indices.dict', 'rb'))
            sampling_probs = pickle.load(open('data/sampling_probs.list', 'rb'))
        else:
            idx_pairs, noise_dist, vocab_idx, ctx_indices, sampling_probs \
                    = build_skip_gram_dataset(docs_filename, vocab_filename, window_size)
            pickle.dump(idx_pairs, open('data/idx_pairs.list', 'wb'))
            pickle.dump(noise_dist, open('data/noise_samples.list', 'wb'))
            pickle.dump(vocab_idx, open('data/vocab_idx.dict', 'wb'))
            pickle.dump(ctx_indices, open('data/ctx_indices.dict', 'wb'))
            pickle.dump(sampling_probs, open('data/sampling_probs.list', 'wb'))

        assert len(idx_pairs) == len(sampling_probs)

        self.idx_pairs = np.array(idx_pairs)
        self.noise_dist = noise_dist
        self.vocab_idx = vocab_idx
        self.ctx_indices = ctx_indices
        self.neg_samples = neg_samples
        self.sampling_probs = sampling_probs

        print("The number of samples :", self.idx_pairs.shape[0])

    def get_negatives(self, ctxs):
        noise = random.sample(self.noise_dist, self.neg_samples)
        while np.intersect1d(np.array(noise), np.array(ctxs)).shape[0] != 0:
            noise = random.sample(self.noise_dist, self.neg_samples)

        return np.array(noise)

    def __len__(self):
        return self.idx_pairs.shape[0]
    
    def __getitem__(self, idx):
        center_word = torch.LongTensor(np.array([self.idx_pairs[idx,0]]))
        context_word = torch.LongTensor(np.array([self.idx_pairs[idx,1]]))

        noise = self.get_negatives(self.ctx_indices[self.idx_pairs[idx,0]])
        noise = torch.LongTensor(noise)

        return center_word, context_word, noise


class SkipGram(nn.Module):
    """
    Implementation of https://proceedings.neurips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf
    """
    def __init__(self, num_embeddings, embedding_dim, padding_idx=1):
        super(SkipGram, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.w_i = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim, padding_idx=padding_idx)
        self.w_o = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim, padding_idx=padding_idx)
        self.log_sigmoid = nn.LogSigmoid()

        self.init_weights()

    def init_weights(self):
        init_range = (2.0 / (self.num_embeddings + self.embedding_dim)) ** 0.5 # Xavier initialization
        self.w_i.weight.data.uniform_(-init_range, init_range)
        self.w_o.weight.data.uniform_(-0, 0)
    
    def forward(self, cent, ctx, noise):
        v = self.w_i(cent)  # (B, 1, embedding_dim)
        v_prime = self.w_o(ctx)  # (B, embedding_dim)

        # positive value
        pos_val = self.log_sigmoid(torch.sum(v_prime*v, dim=1)).squeeze()  # (B)
        v_hat = self.w_o(noise)  # (B, neg, embedding_dim)

        # negative values
        neg_val = torch.bmm(v_hat, v.unsqueeze(2)).squeeze(2) # (B, neg)
        neg_val = self.log_sigmoid(-torch.sum(neg_val, dim=1)).squeeze()

        loss = pos_val + neg_val

        return -loss.mean()


def train_skip_gram(total_step=300000, batch_size=128):
    # device = torch.cuda.device('cuda:0')

    dataset = SkipGramDataset('data/nate_pann_ranking20200901_20210628_rm_noisy.json', 'data/vocab_freq.js')
    sampler = torch.utils.data.WeightedRandomSampler(dataset.sampling_probs, batch_size, replacement=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, sampler=sampler)
    model = SkipGram(dataset.vocab_size, 512).cuda()

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    optimizer = torch.optim.AdamW(model.parameters(), 0.001, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_step)

    model.train()
    should_keep_training = True
    it = 0
    while should_keep_training:
        for b, (center, ctx, noise) in enumerate(dataloader):
            center = center.cuda().squeeze(1)
            ctx = ctx.cuda().squeeze(1)
            noise = noise.cuda()

            optimizer.zero_grad()

            loss = model(center, ctx, noise)
            loss.backward()

            optimizer.step()
            scheduler.step()
            it += 1

            print('\rIter %d loss: %.4f' % (it, loss.detach().item()), end='')
        

            if it % 20000 == 0:
                torch.save(model.state_dict(), 'checkpoints/skip_gram_it%d_loss%.4f.pth' % (it, loss.detach().item()))

        
if __name__=='__main__':
    train_skip_gram()