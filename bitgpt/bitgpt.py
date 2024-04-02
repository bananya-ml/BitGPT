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
            y = self.lowbit_gemm_kernel(x_quant, w) / (w_scale * x_scale)
            if self.bias is not None:
                y += self.bias
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
    
    def lowbit_gemm_kernel(self, x, w):
        '''
        Low-bit GEMM kernel for 2-bit weights.
        Args:
            x (torch.Tensor): Input tensor with shape [n, d].
            w (torch.Tensor): Weight tensor with shape [d, k], quantized to 2-bit values (-1, 0, 1, 2).
        Returns:
            y (torch.Tensor): Output tensor with shape [n, k].
        '''
        # compute the GEMM using bit operations
        # we split the weight tensor into four tensors based on the value
        w_neg1 = (w == -1).float()
        w_0 = (w == 0).float()
        w_1 = (w == 1).float()

        # compute the partial products
        y_neg1 = x @ w_neg1.transpose(0, 1)
        y_0 = x @ w_0.transpose(0, 1)
        y_1 = x @ w_1.transpose(0, 1)

        # combine the partial products
        y = y_1 - y_neg1

        return y

class Head(nn.Module):
    ''' 
    One head of self-attention.

    Args:
        config: the model configuration
        head_size: the size of the head

    Attributes:
        key: the bitlinear layer for the key
        query: the bitlinear layer for the query
        value: the bitlinear layer for the value
        tril: a lower triangular matrix of ones
        dropout: a dropout layer

    Return:
        out: the output of the head
    '''

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
    ''' 
    A simple linear layer followed by a non-linearity 
    
    Args:
        config: the model configuration
    
    Attributes:
        net: a sequence of layers

    Return:
        out: the output of the feedforward network
    '''

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
    ''' 
    Multiple heads of self-attention in parallel 
    
    Args:
        config: the model configuration
        head_size: the size of the head
    
    Attributes:
        heads: a list of heads
        proj: a bitlinear layer
        dropout: a dropout layer

    Return:
        out: the output of the multi-head attention
    '''

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
    ''' 
    Transformer block: communication followed by computation 
    
    Args:
        config: the model configuration

    Attributes:
        sa: the multi-head self-attention layer
        ffwd: the feedforward layer

    Return:
        out: the output of the block
    '''

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
        self.out_norm = nn.LayerNorm(config.n_embd)
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
        x = self.out_norm(x)
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
        '''
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Args:
            idx: the initial seed indices (LongTensor of shape (b,t))
            max_new_tokens: the number of tokens to generate
            temperature: the temperature of the sampling distribution
            top_k: the number of top-k most likely tokens to sample from. If None, sample from the whole distribution.
        
        Returns:
            idx: the tensor of generated indices (LongTensor of shape (b,t+max_new_tokens))
        '''
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
        '''
        Configure optimizer (AdamW) for training the model.

        Args:
            weight_decay (float): the weight decay coefficient
            learning_rate (float): the learning rate
            betas (tuple): the beta coefficients for AdamW
            device_type (str): the device type
        
        Returns:
            optimizer: the AdamW optimizer
        '''
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