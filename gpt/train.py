import torch 
import tiktoken
import wandb
from gpt import GPTLanguageModel, GPTConfig
from tqdm import tqdm
import os
import math
import argparse

def ParseArgs():
    parser = argparse.ArgumentParser(description='GPT model')
    parser.add_argument('--batch-size',type=int,default=64,metavar='N',
                        help='batch size for training(default: 64)')
    parser.add_argument('--block-size',type=int,default=256,metavar='N',
                        help='maximum context length for predictions(default: 256)')
    parser.add_argument('--max-iters',type=int,default=500000,metavar='N',
                        help='number of epochs to train(default: 500000)')
    parser.add_argument('--eval-iters',type=int,default=200,metavar='N',
                        help='number of batches used to estimate loss during eval(default: 200)')
    parser.add_argument('--eval-interval',type=int,default=2000,metavar='N',
                        help='interval after which eval is performed(default: 2000)')                        
    parser.add_argument('--lr',type=float,default=6e-4,metavar='LR',
                        help='learning rate(default: 6e-4)')
    parser.add_argument('--n-head',type=float,default=4,metavar='M',
                        help='number of heads in the transformer architecture(default: 4)')
    parser.add_argument('--n-layer',type=float,default=4,metavar='M',
                        help='number of layers of the transformer architecture(default: 4)')
    parser.add_argument('--n-embd',type=float,default=384,metavar='M',
                        help='embedding dimension(default: 384)')
    parser.add_argument('--dropout', '--d',type=float,default=0.2,metavar='S',
                        help='dropout value(default: 0.2)')
    parser.add_argument('--weight-decay','--wd',type=float,default=1e-1,metavar='WD',
                        help='weight decay(default: 1e-1)')
    parser.add_argument('--decay-lr',type=bool,default=True,metavar='S',
                        help='flag for learning rate decay(default: True)')
    parser.add_argument('--warmup-iters',type=int,default=200,metavar='S',
                        help='steps to warmup lr decay(default: 200)')
    parser.add_argument('--lr-decay-iters',type=int,default=500000,metavar='S',
                        help='should be ~= max_iters per Chinchilla(default: 500000)')
    parser.add_argument('--min-lr',type=int,default=6e-5,metavar='S',
                        help='should be learning rate/10 per Chinchilla(default: 6e-5)')
    parser.add_argument('--wandb-log',type=bool,default=False,metavar='S',
                        help='logging using wandb(default: False)')
    parser.add_argument('--seed',type=int,default=1337,metavar='S',
                        help='random seed(default: 1337)')
    
    args = parser.parse_args()
    return args

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

# data loading

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
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


args = ParseArgs()

# hyperparameters
batch_size = 12  # how many independent sequences will we process in parallel?
block_size = 128  # what is the maximum context length for predictions?
eval_interval = 2000
eval_iters = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_type = 'cuda' if 'cuda' in device else 'cpu'
print(f"Using {device} for training")

# model
n_embd = 256
n_head = 4
n_layer = 4
dropout = 0.2

# optimizer
max_iters = 50000
learning_rate = 6e-4
beta1 = 0.9
beta2= 0.95
weight_decay = 1e-1

#for lr scheduler
decay_lr = True
warmup_iters = 200
lr_decay_iters = 50000
min_lr = 6e-5


best_val_loss = 1e9
inference = False

#wandb logging
wandb_log = False
wandb_project = 'GPT'
# ------------

torch.manual_seed(1337)

with open(os.path.join(os.getcwd(),'data/astro.txt'), encoding='utf-8') as f:
    text = f.read()

vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency

'''
chars = sorted(list(set(text)))
vocab_size = len(chars) # for individual characters as elements

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


# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  vocab_size=vocab_size, dropout=dropout)

config = GPTConfig(**model_args)
model = GPTLanguageModel(config)
m = model.to(device)

# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

if wandb_log:
     wandb.init(project=wandb_project, config=config)

for iter in tqdm(range(max_iters)):

    lr = get_lr(iter) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        if wandb_log:
             wandb.log({
                  "iter": iter,
                  "train/loss": losses['train'],
                  "val/loss": losses['val'],
                  "lr": lr
             })

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

checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
torch.save(checkpoint, os.path.join(os.getcwd(), "models/astrogpt.pt"))