import torch
import torch.nn as nn
from torch.nn import functional as F



class Head(nn.Module):
    """"" one head of self-attention """

    def __init__(self, n_embd, head_size, block_size, dropout = 0.0) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # computer scores
        wei = q @ k.transpose(-2,-1) * C ** -0.5 # (B,T,C) @ (B,C,T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        # weighted aggregation of the values
        v = self.value(x) #(B,T,C)
        out = wei @ v #(B,T,C)
        
        return out
    



class MultiHeadAttention(nn.Module):

    def __init__(self, n_embd, num_heads, head_size, block_size, dropout = 0.0) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)  # projection is needed because of the residue connection
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):

    def __init__(self, n_embd, dropout = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)
    



class Block(nn.Module):

    def __init__(self,  n_embd, n_head, block_size, dropout = 0.0) -> None:
        super().__init__()
        head_size = n_embd//n_head
        self.sa = MultiHeadAttention(n_embd, n_head, head_size, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        return x
    


class GPTLanguageModel(nn.Module):

    def __init__(self, vocab_size, n_embd, n_head, block_size, n_layer = 6, dropout = 0.0):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.block_size = block_size


        self.blocks = nn.Sequential(*[Block(n_embd, n_head = n_head, block_size = self.block_size, dropout = dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)

        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        tok_emb = self.token_embedding_table(idx) #(B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device = device)) # (T,C)
        x = tok_emb + pos_emb #(B,T,C)
        x = self.blocks(x) # (B,T,C)
        logits = self.lm_head(x) #(B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]

            # get the predictions
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] #  only use last token, (B,C)
            probs = F.softmax(logits, dim = -1) # (B,C)

            idx_next = torch.multinomial(probs, num_samples = 1) #(B,1)
            idx = torch.cat((idx, idx_next), dim = 1)
        
        return idx