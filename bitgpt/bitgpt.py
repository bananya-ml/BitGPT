import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import inspect

@dataclass
class BitGPTConfig:
    block_size: int = 128
    vocab_size: int = 50304
    n_embd: int = 384
    n_head: int = 2
    n_layer: int = 2
    dropout: float = 0.2


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps:float=1e-6):
        '''
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.
        '''
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x):
        '''
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.
        '''
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        '''
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.
        '''
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    
class BitLinear(nn.Linear):
    '''
    '''
    def __init__(self, in_features: int, out_features: int, bias=False, inference=False):
        super(BitLinear, self).__init__(in_features, out_features, bias)
        self.dim = self.in_features
        self.inference = inference
        self.out_features = self.out_features
        self.rms_norm = RMSNorm(self.dim)

    def forward(self, x):
        '''
        Args:
            x: an input tensor with shape [n, d]
        
        Returns:
            y: an output tensor with shape [n, d]
        '''
        w = self.weight
        if not self.inference:
            x_norm = self.rms_norm(x)
            # A trick for implementing Straight−Through−Estimator (STE) using detach()
            x_quant = x_norm + (self.activation_quant(x_norm) - x_norm).detach()
            w_quant = w + (self.weight_quant(w) -w).detach()
            y = F.linear(x_quant, w_quant)
            return y
        
        else:
            x_quant, x_scale = self.activation_norm_quant(x)
            w_scale = self.weight_scale
            # according to the paper, this linear layer may have to be replaced by a gemm_lowbit_kernel,
            # but no such kernel is available, nor any directions on how to implement it, so we'll just use linear
            y = F.linear(x_quant, w.float(), self.bias) / (x_scale * w_scale)
            return y

    def activation_quant(self, x):
        '''
        Per−token quantization to 8 bits. No grouping is needed for quantization.
        
        Args:
            x: an activation tensor with shape [n, d]
        
        Returns:
            y: a quantized activation tensor with shape [n, d]
        '''
        scale = 127.0/x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        y = (x*scale).round().clamp_(-128, 127)/scale
        return y
    
    def activation_norm_quant(self, x):
        '''
        
        '''
        x = self.rms_norm(x)
        scale = 127./x.abs().max(dim=-1,keepdim=True).values.clamp_(min=1e-5)
        y= (x * scale).round().clamp_(-128,127)
        return y, scale
    
    def weight_quant(self, w):
        ''' 
        Per−tensor quantization to 1.58 bits. No grouping is needed for quantization.
        
        Args:
            w: a weight tensor with shape [d, k]
        
        Returns:
            u: a quantized weight with shape [d, k]
        '''
        scale = 1.0/w.abs().mean().clamp_(min=1e-5)
        u = (w*scale).round().clamp_(-1,1)/scale
        return u


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, config, head_size):
        super().__init__()
        self.key = BitLinear(config.n_embd, head_size, bias=False)
        self.query = BitLinear(config.n_embd, head_size, bias=False)
        self.value = BitLinear(config.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(
            torch.ones(config.block_size, config.block_size)))

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            BitLinear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            BitLinear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)
    
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, config, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(config, head_size) for _ in range(config.n_head)])
        self.proj = BitLinear(head_size * config.n_head, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, config):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.sa = MultiHeadAttention(config, head_size)
        self.ffwd = FeedFoward(config)

    def forward(self, x):
        x = x + self.sa(x)
        x = x + self.ffwd(x)
        return x

class BitGPTLanguageModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_layer)])
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        
        B, T = idx.shape
        device = idx.device
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer