import torch 
import tiktoken
import wandb
from bitgpt import BitGPTLanguageModel, BitGPTConfig, BitLinear
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
                        help='number of epoch to train(default: 500000)')
    parser.add_argument('--eval_iters',type=int,default=200,metavar='N',
                        help='number of batches used to estimate loss during eval(default: 200)')
    parser.add_argument('--eval_interval',type=int,default=2000,metavar='N',
                        help='interval after which eval is performed(default: 2000)')                        
    parser.add_argument('--lr',type=float,default=6e-4,metavar='LR',
                        help='learning rate(default: 1e-3)')
    parser.add_argument('--n-head',type=float,default=4,metavar='M',
                        help='number of heads in the transformer architecture(default: 4)')
    parser.add_argument('--n-layer',type=float,default=4,metavar='M',
                        help='number of layers of the transformer architecture(default: 4)')
    parser.add_argument('--n-embd',type=float,default=384,metavar='M',
                        help='embedding dimension(default: 384)')
    parser.add_argument('--dropout',type=float,default=0.2,metavar='S',
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
    parser.add_argument('--wandb_log',type=bool,default=False,metavar='S',
                        help='logging using wandb(default: False)')
    parser.add_argument('--seed',type=int,default=1337,metavar='S',
                        help='random seed(default: 1337)')
    parser.add_argument('--log-interval',type=int,default=100,metavar='N',
                        help='how many batches to wait before logging training status')
    
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


args = ParseArgs()

# hyperparameters
batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 256  # what is the maximum context length for predictions?
eval_interval = 2000
eval_iters = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_type = 'cuda' if 'cuda' in device else 'cpu'
print(f"Using {device} for training")

# model
n_embd = 64
n_head = 1
n_layer = 1
dropout = 0.2

# optimizer
max_iters = 500000
learning_rate = 6e-4
beta1 = 0.9
beta2= 0.95
weight_decay = 1e-1

#for lr scheduler
decay_lr = True # whether to decay the learning rate
warmup_iters = 200 # how many steps to warm up for
lr_decay_iters = 500000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla


best_val_loss = 1e9
inference = False

#wandb logging
wandb_log = False
wandb_project = 'bitGPT'
# ------------

torch.manual_seed(1337)

with open(os.path.join(os.getcwd(),'data\\shakespeare.txt'), 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))

vocab_size = len(chars) # for individual characters as elements
#vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency

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
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  vocab_size=vocab_size, dropout=dropout)

config = BitGPTConfig(**model_args)
model = BitGPTLanguageModel(config)
m = model.to(device)

# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

if wandb_log:
     wandb.init(project=wandb_project, config=config)

for iter in tqdm(range(max_iters)):

    lr = get_lr(iter) if decay_lr else learning_rate

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
#torch.save(checkpoint, os.path.join(os.getcwd(), "models\\bitGPT.pt"))