from gpt import GPTLanguageModel, GPTConfig
from contextlib import nullcontext
import torch
import os

# encoder: take a string, output a list of integers
def encode(s): return [stoi[c] for c in s]
# decoder: take a list of integers, output a string
def decode(l): return ''.join([itos[i] for i in l])


num_samples = 2
max_new_tokens = 500
temperature = 1.0
top_k = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_type = 'cuda' if 'cuda' in device else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)



ckpt = torch.load(os.path.join(os.getcwd(), "models\\gpt.pt"))

config = GPTConfig(**ckpt['model_args'])
model = GPTLanguageModel(config)
state_dict = ckpt['model']
model.load_state_dict(state_dict)

model.eval()
model.to(device)

prompt = ''

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

start_ids = torch.zeros((1), dtype=torch.long, device=device) if prompt == '' else encode(prompt) 
x = (torch.tensor(start_ids,dtype=torch.long, device=device)[None, ...])

with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')
