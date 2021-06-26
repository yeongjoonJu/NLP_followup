import torch
import torch.nn as nn

class ManualEmbedding(nn.Module):
    def __init__(self, vocab_size, out_dim, unk=0):
        super(ManualEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.unk = unk
        self.projector = nn.Parameter(torch.FloatTensor(vocab_size, out_dim), requires_grad=True)
    
    def forward(self, x):
        x[x >= self.vocab_size] = self.unk
        return self.projector[x]

class CBOW(nn.Module):
    """Continuous Bag of Words
        x [B, Window]
    """
    def __init__(self, num_embeddings, embedding_dim, padding_idx=1):
        super(CBOW, self).__init__()
        self.embedding_dim = embedding_dim
        self.w = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim, padding_idx=padding_idx)
        self.w_p = nn.Linear(embedding_dim, num_embeddings)

    def forward(self, x):
        assert len(x.shape) == 2
        b = x.shape[0]
        x = x.view(-1)
        x = self.w(x)
        x = x.view(b, -1, self.embedding_dim)
        x = torch.mean(x, dim=1)
        return self.w_p(x)

class SkipGram(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=1, window_size=2):
        super(SkipGram, self).__init__()
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.w = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim, padding_ix=padding_idx)
        w_p = []
        for _ in range(window_size*2):
            w_p.append(nn.Linear(embedding_dim, num_embeddings))
        
        self.w_p = nn.ModuleList(w_p)
    
    def forward(self, x):
        x = self.w(x)
        out = []
        for i in range(self.window_size*2):
            out.append(self.w_p[i](x))
        
        return out


if __name__=='__main__':
    # Training CBOW
    