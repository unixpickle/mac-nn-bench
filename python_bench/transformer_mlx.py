import time
import math
import mlx.nn as nn
import mlx.core as mx
from dataclasses import dataclass
import mlx.optimizers as optim
from functools import partial
import mlx.utils

def main():
    config = TransformerConfig()
    model = Transformer(config)
    batch = mx.random.randint(0, config.vocab_size, [4, config.token_count])
    optimizer = optim.Adam(learning_rate=0.001)

    state = [model.state, optimizer.state]

    def compute_loss(model, inp):
        output = model(inp)
        loss = mx.mean(nn.losses.cross_entropy(output, inp))
        return loss

    @partial(mx.compile, inputs=state, outputs=state)
    def step(inp):
        loss_fn = nn.value_and_grad(model, compute_loss)
        loss, grads = loss_fn(model, inp)
        optimizer.update(model, grads)
        return loss

    print("warming up...")
    for _ in range(5):
        step(batch)
    mlx.core.eval(model.parameters())

    print('benchmarking...')
    t1 = time.time()
    num_its = 20
    for _ in range(num_its):
        step(batch)
    mlx.core.eval(model.parameters())
    elapsed = (time.time() - t1) / num_its

    num_params = sum(x[1].size for x in mlx.utils.tree_flatten(model.parameters()))
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
        theta = (math.log(base) * mx.arange(0, dim//2, dtype=mx.float32) / dim).exp()
        indices = mx.arange(0, max_tokens, dtype=mx.float32)[:, None]
        args = indices * theta
        self.cache = mx.stack([args.cos(), args.sin()], axis=-1)
    
    def __call__(self, x: mx.array) -> mx.array:
        x2d = x.reshape(x.shape[:3] + (x.shape[3] // 2, 2)) # [B x H x T x C/2 x 2]
        shapedCache = self.cache.reshape(x2d.shape[2], x2d.shape[3], 2)
        x0 = x2d[..., 0]
        x1 = x2d[..., 1]
        r0 = shapedCache[..., 0]
        r1 = shapedCache[..., 1]
        return mx.stack([x0*r0 - x1*r1, x0*r1 + x1*r0], axis=-1).flatten(start_axis=3)

class Attention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.rope = RoPE(dim=config.head_dim, max_tokens=config.token_count)
        
        self.q_proj = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.k_proj = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.v_proj = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.out_proj = nn.Linear(config.model_dim, config.model_dim, bias=False)
        
        causal_mask = mx.tril(mx.full((config.token_count, config.token_count), float('-inf')))
        self.causal_mask = causal_mask

    def __call__(self, x):
        # Project to query, key, and value spaces
        q = self.q_proj(x) / (self.config.head_dim ** 0.25)
        k = self.k_proj(x) / (self.config.head_dim ** 0.25)
        v = self.v_proj(x)

        # Shape [B x T x C] -> [B x H x T x C/H]
        q = self.move_heads_to_outer(q)
        k = self.move_heads_to_outer(k)
        v = self.move_heads_to_outer(v)

        # Apply RoPE to query and key
        q, k = self.rope(q), self.rope(k)

        # Attention calculation with causal masking
        energy = mx.einsum('bhtd,bhld->bhtl', q, k)
        energy = energy + self.causal_mask
        attention = mx.softmax(energy, axis=-1)
        
        # Apply attention weights to value and move heads back
        out = mx.einsum('bhtl,bhld->bhtd', attention, v)
        return self.out_proj(self.move_heads_to_inner(out))

    def move_heads_to_outer(self, x: mx.array) -> mx.array:
        b, t, _c = x.shape
        h = self.config.num_heads
        head_dim = self.config.head_dim
        return x.reshape(b, t, h, head_dim).swapaxes(1, 2)  # [B x H x T x C/H]

    def move_heads_to_inner(self, x: mx.array) -> mx.array:
        b, h, t, head_dim = x.shape
        return x.swapaxes(1, 2).reshape(b, t, h * head_dim)  # [B x T x C]

class Block(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.attn = Attention(config)
        self.norm1 = nn.LayerNorm(config.model_dim)
        self.norm2 = nn.LayerNorm(config.model_dim)
        self.lin1 = nn.Linear(config.model_dim, config.model_dim * 2, bias=False)
        self.lin2 = nn.Linear(config.model_dim * 2, config.model_dim, bias=False)

    def __call__(self, x):
        # Apply normalization, attention, and residual connection
        h = x + self.attn(self.norm1(x))
        
        # Feed-__call__ network with GELU and residual connection
        h = h + self.lin2(self.gelu(self.lin1(self.norm2(h))))
        
        return h

    def gelu(self, x):
        return nn.GELU()(x)

class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Embedding and Transformer layers
        self.embed = nn.Embedding(config.vocab_size, config.model_dim)
        self.layers = nn.Sequential(*[Block(config) for _ in range(config.layer_count)])
        self.norm_out = nn.LayerNorm(config.model_dim)
        
        # Unembedding layer for output probabilities
        self.unembed = nn.Linear(config.model_dim, config.vocab_size, bias=False)

    def __call__(self, x):
        # Embed input tokens
        h = self.embed(x)  # [N x T x D]
        
        # Pass through each Transformer layer
        h = self.layers(h)
        
        # Normalize and unembed for final output
        h = self.norm_out(h)
        return self.unembed(h)

if __name__ == '__main__':
    main()
