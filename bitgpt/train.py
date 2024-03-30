import torch 
import tiktoken
from bitgpt import BitGPTLanguageModel, BitGPTConfig, BitLinear
from tqdm import tqdm
import os
import math

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

def quantize(w: torch.Tensor):
    """
    Offline quantization of a set of weights to int8 based on the mean of the absolute values.

    This operation casts the weights to int8.
    Args:
        w (torch.Tensor): weights
    Returns:
        w_quant (torch.Tensor): quantized weights
        scale (torch.Tensor): scale factor
    """
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    w_quant = (w * scale).round().clamp_(-1,1).to(torch.int8)
    w_quant.requires_grad = False
    return w_quant, scale

# data loading


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
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

# hyperparameters
batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 128  # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} for training")
eval_iters = 200
n_embd = 384
n_head = 4
n_layer = 4
dropout = 0.2

#for lr scheduler
decay_lr = True # whether to decay the learning rate
warmup_iters = 200 # how many steps to warm up for
lr_decay_iters = 5000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# ------------

torch.manual_seed(1337)

with open(os.path.join(os.getcwd(),'data\\shakespeare.txt'), 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))

vocab_size = len(chars) # for individual characters as elements
#vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# encoder: take a string, output a list of integers
def encode(s): return [stoi[c] for c in s]
# decoder: take a list of integers, output a string
def decode(l): return ''.join([itos[i] for i in l])

'''
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)
'''

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

inference = False

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  vocab_size=vocab_size, dropout=dropout)

config = BitGPTConfig(**model_args)

model = BitGPTLanguageModel(config)
m = model.to(device)

# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

best_val_loss = 1e9

for iter in tqdm(range(max_iters)):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

meta = {} # save full precision weight scales for generation later

for name, layer in model.named_modules():
            if isinstance(layer, BitLinear):
                for k, v in layer.state_dict().items():
                    if 'weight' in k and 'norm' not in k:
                        w_quant, scale = quantize(v)
                        layer.weight.requires_grad = False
                        layer.weight.data = w_quant
                        layer.weight_scale = scale
                        meta[name] = scale
checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'best_val_loss': best_val_loss,
                    'config': config,
                    'meta': meta
                }
torch.save(checkpoint, os.path.join(os.getcwd(), "models\\bitGPT.pt"))
# open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))