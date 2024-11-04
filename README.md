# 🍎 mac-nn-bench

Neural network benchmarks for Mac-native deep learning frameworks.

# Current results

These are results on an M2 Max with 32 GB of RAM

| Benchmark   | MLX          | PyTorch      | Honeycrisp   | Honeycrisp + ANE |
|-------------|--------------|--------------|--------------|------------------|
| Transformer | 5.843 TFLOPs | 4.251 TFLOPs | 3.618 TFLOPs | 4.031 TFLOPs     |

# Running benchmarks

## Transformer

```bash
$ python -m python_bench.transformer_mlx
$ python -m python_bench.transformer_torch
$ swift run -c release Transformer
$ swift run -c release TransformerANE
```
