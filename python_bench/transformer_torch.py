import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataclasses import dataclass

device = torch.device("mps")


def main():
    config = TransformerConfig()
    model = Transformer(config).to(device)
    batch = torch.randint(0, config.vocab_size, (4, config.token_count)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    def compute_loss(model, inp):
        output = model(inp)
        loss = nn.CrossEntropyLoss()(output.view(-1, config.vocab_size), inp.view(-1))
        return loss

    def step(inp):
        model.train()
        optimizer.zero_grad()
        loss = compute_loss(model, inp)
        loss.backward()
        optimizer.step()
        return loss.item()

    print("warming up...")
    for _ in range(5):
        step(batch)
    torch.mps.synchronize()

    print("benchmarking...")
    num_its = 20
    t1 = time.time()
    for _ in range(num_its):
        step(batch)
    torch.mps.synchronize()
    elapsed = (time.time() - t1) / num_its

    num_params = sum(p.numel() for p in model.parameters())
    num_flops = 6 * len(batch) * num_params * config.token_count
    gflops = num_flops / (elapsed * 1e9)
    print(f'gflops: {gflops}')



@dataclass
class TransformerConfig:
    vocab_size: int = 16384 + 256
    token_count: int = 128 + 256
    layer_count: int = 24
    model_dim: int = 1024
    head_dim: int = 64

    @property
    def num_heads(self) -> int:
        return self.model_dim // self.head_dim

class RoPE(nn.Module):
    def __init__(self, dim: int, max_tokens: int, base: int = 10000):
        super().__init__()
        theta = torch.exp((math.log(base) * torch.arange(0, dim//2, dtype=torch.float32) / dim))
        indices = torch.arange(0, max_tokens, dtype=torch.float32).unsqueeze(1)
        args = indices * theta
        self.cache = torch.stack([args.cos(), args.sin()], dim=-1).to(device)
    
    def forward(self, x):
        x2d = x.reshape(x.shape[:3] + (x.shape[3] // 2, 2))  # [B x H x T x C/2 x 2]
        shaped_cache = self.cache[:x2d.shape[2], :x2d.shape[3]].to(device)
        x0, x1 = x2d[..., 0], x2d[..., 1]
        r0, r1 = shaped_cache[..., 0], shaped_cache[..., 1]
        return torch.stack([x0 * r0 - x1 * r1, x0 * r1 + x1 * r0], dim=-1).flatten(start_dim=3)

class Attention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.rope = RoPE(dim=config.head_dim, max_tokens=config.token_count)
        
        self.q_proj = nn.Linear(config.model_dim, config.model_dim, bias=False).to(device)
        self.k_proj = nn.Linear(config.model_dim, config.model_dim, bias=False).to(device)
        self.v_proj = nn.Linear(config.model_dim, config.model_dim, bias=False).to(device)
        self.out_proj = nn.Linear(config.model_dim, config.model_dim, bias=False).to(device)
        
        causal_mask = torch.tril(torch.full((config.token_count, config.token_count), float('-inf')))
        self.causal_mask = causal_mask.to(device)

    def forward(self, x):
        # Project to query, key, and value spaces
        q = self.q_proj(x) / (self.config.head_dim ** 0.25)
        k = self.k_proj(x) / (self.config.head_dim ** 0.25)
        v = self.v_proj(x)

        # Shape [B x T x C] -> [B x H x T x C/H]
        q, k, v = self.move_heads_to_outer(q), self.move_heads_to_outer(k), self.move_heads_to_outer(v)

        # Apply RoPE to query and key
        q, k = self.rope(q), self.rope(k)

        # Attention calculation with causal masking
        energy = torch.einsum('bhtd,bhld->bhtl', q, k) + self.causal_mask
        attention = torch.softmax(energy, dim=-1)
        
        # Apply attention weights to value and move heads back
        out = torch.einsum('bhtl,bhld->bhtd', attention, v)
        return self.out_proj(self.move_heads_to_inner(out))

    def move_heads_to_outer(self, x):
        b, t, _c = x.shape
        h = self.config.num_heads
        head_dim = self.config.head_dim
        return x.reshape(b, t, h, head_dim).transpose(1, 2)  # [B x H x T x C/H]

    def move_heads_to_inner(self, x):
        b, h, t, head_dim = x.shape
        return x.transpose(1, 2).reshape(b, t, h * head_dim)  # [B x T x C]

class Block(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attn = Attention(config)
        self.norm1 = nn.LayerNorm(config.model_dim).to(device)
        self.norm2 = nn.LayerNorm(config.model_dim).to(device)
        self.lin1 = nn.Linear(config.model_dim, config.model_dim * 2, bias=False).to(device)
        self.lin2 = nn.Linear(config.model_dim * 2, config.model_dim, bias=False).to(device)

    def forward(self, x):
        h = x + self.attn(self.norm1(x))
        h = h + self.lin2(self.gelu(self.lin1(self.norm2(h))))
        return h

    def gelu(self, x):
        return F.gelu(x)

class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.model_dim).to(device)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.layer_count)])
        self.norm_out = nn.LayerNorm(config.model_dim).to(device)
        self.unembed = nn.Linear(config.model_dim, config.vocab_size, bias=False).to(device)

    def forward(self, x):
        h = self.embed(x.to(device))
        for layer in self.layers:
            h = layer(h)
        h = self.norm_out(h)
        return self.unembed(h)

if __name__ == '__main__':
    main()
