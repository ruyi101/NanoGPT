import torch
from GPTmodel import GPTLanguageModel
from Tokenizer import tokenizer

# train test split
def train_test_split(data, train_ratio=0.9):
    data = torch.tensor(data, dtype = torch.long)
    n = int(train_ratio * len(data))

    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data





# data loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i+block_size] for i in ix])
    y = torch.stack([data[i+1 : i+block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)

    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out




batch_size = 128
block_size = 64
max_iters = 50000
eval_interval = 5000
learning_rate = 2 * 1e-5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
vocab_size = 768

## train tokenizer

with open('data/shakespeare.txt', 'r', encoding = 'utf-8') as f:
    shakespeare = f.read()


tok = tokenizer()
tok.train(shakespeare, vocab_size)

## prepare data
with open('data/Hemingway.txt', 'r', encoding = 'utf-8') as f:
    text = f.read()


data = tok.encode(text)
train_data, val_data = train_test_split(data)




model = GPTLanguageModel(vocab_size, n_embd, n_head, block_size, n_layer, dropout = dropout)
m = model.to(device)
# num of paras
print(sum(p.numel() for p in m.parameters())/1e6,  'M parameters')
# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)


for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']: .4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(tok.decode(m.generate(context, max_new_tokens = 1000)[0].tolist()))



