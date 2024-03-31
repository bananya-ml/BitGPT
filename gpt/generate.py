from gpt import GPTLanguageModel, GPTConfig
from contextlib import nullcontext
import torch
import os
import argparse

def ParseArgs():
    parser = argparse.ArgumentParser(description='GPT model')
    parser.add_argument('--prompt',type=str,default='',metavar='N',
                        help='generation from the model follows the prompt(default: '')')
    parser.add_argument('--num-samples',type=int,default=2,metavar='N',
                        help='number of samples to generate(default: 2)')
    parser.add_argument('--max-new-tokens',type=int,default=2000,metavar='N',
                        help='maximum context length for predictions(default: 2000)')
    parser.add_argument('--temperature',type=int,default=1,metavar='N',
                        help='1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions(default= 1.0)')
    parser.add_argument('--top-k',type=int,default=200,metavar='N',
                        help='# retain only the top_k most likely tokens, clamp others to have 0 probability(default: 200)')
    args = parser.parse_args()
    return args

# encoder: take a string, output a list of integers
def encode(s): return [stoi[c] for c in s]
# decoder: take a list of integers, output a string
def decode(l): return ''.join([itos[i] for i in l])

args = ParseArgs()

prompt = args.prompt
num_samples = args.num_samples
max_new_tokens = args.max_new_tokens
temperature = args.temperature
top_k = args.top_k
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} for generation.")
device_type = 'cuda' if 'cuda' in device else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)



ckpt = torch.load(os.path.join(os.getcwd(), "models\\gpt.pt"))

config = GPTConfig(**ckpt['model_args'])
model = GPTLanguageModel(config)
print(f"Loaded model {model.__class__.__name__}")
state_dict = ckpt['model']
model.load_state_dict(state_dict)

model.eval()
model.to(device)

prompt = ''

with open(os.path.join(os.getcwd(),'data\\shakespeare.txt'), 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

start_ids = torch.zeros((1), dtype=torch.long, device=device) if prompt == '' else encode(prompt) 
x = start_ids.clone().detach()[None, ...]

with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')
            with open(os.path.join(os.getcwd(),'gpt\\more.txt'),'w') as file:
                file.write(decode(y[0].tolist()))
