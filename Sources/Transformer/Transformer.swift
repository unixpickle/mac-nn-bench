import Cocoa
import Foundation
import Honeycrisp

struct TransformerConfig {
  var VocabSize: Int
  let TokenCount: Int
  var WeightGradBackend: Backend
  var LayerCount: Int = 24
  var ModelDim: Int = 1024
  var HeadDim: Int = 64
}

/// Implementation based on
/// https://pytorch.org/torchtune/stable/_modules/torchtune/modules/position_embeddings.html#RotaryPositionalEmbeddings
class RoPE {
  let cache: Tensor

  init(dim: Int, maxTokens: Int, base: Int = 10000, dtype: Tensor.DType = .float32) {
    let theta =
      (-log(Float(base)) * Tensor(data: Array(0..<(dim / 2)), shape: [dim / 2], dtype: dtype) / dim)
      .exp()
    let indices = Tensor(data: Array(0..<maxTokens), shape: [maxTokens]).cast(dtype).unsqueeze(
      axis: -1
    ).repeating(
      axis: 1, count: dim / 2)
    let args = indices * theta
    cache = Tensor(stack: [args.cos(), args.sin()], axis: -1)
  }

  func callAsFunction(_ x: Tensor) -> Tensor {
    assert(x.shape.count == 4, "expected [B x H x T x C]")

    let x2D = x.reshape(Array(x.shape[..<3]) + [x.shape[3] / 2, 2])  // [B x H x T x C/2 x 2]
    let shapedCache = cache.reshape([x2D.shape[2], x2D.shape[3], 2])
    let x0 = x2D[..., ..., ..., ..., 0]
    let x1 = x2D[..., ..., ..., ..., 1]
    let r0 = shapedCache[..., ..., 0]
    let r1 = shapedCache[..., ..., 1]
    return Tensor(stack: [x0 * r0 - x1 * r1, x0 * r1 + x1 * r0], axis: -1).flatten(startAxis: 3)
  }
}

class Attention: Trainable {
  let config: TransformerConfig
  let causalMask: Tensor
  var rope: RoPE

  @Child var qProj: Linear
  @Child var kProj: Linear
  @Child var vProj: Linear
  @Child var outProj: Linear

  init(config: TransformerConfig) {
    self.config = config
    rope = RoPE(dim: config.HeadDim, maxTokens: config.TokenCount)
    causalMask = Tensor(constant: 1e8, shape: [config.TokenCount, config.TokenCount]).tril() - 1e8
    super.init()
    self.qProj = Linear(inCount: config.ModelDim, outCount: config.ModelDim, bias: false)
    self.kProj = Linear(inCount: config.ModelDim, outCount: config.ModelDim, bias: false)
    self.vProj = Linear(inCount: config.ModelDim, outCount: config.ModelDim, bias: false)
    self.outProj = Linear(inCount: config.ModelDim, outCount: config.ModelDim, bias: false)
  }

  func callAsFunction(_ x: Tensor) -> Tensor {
    // Go from [B x T x C] -> [B x H x T x C/H]
    func moveHeadsToOuter(_ x: Tensor) -> Tensor {
      x.reshape([x.shape[0], x.shape[1], config.ModelDim / config.HeadDim, config.HeadDim])[
        FullRange(), PermuteAxes(1, 0)]
    }

    // Go from [B x H x T x C/H] -> [B x T x C]
    func moveHeadsToInner(_ x: Tensor) -> Tensor {
      x[FullRange(), PermuteAxes(1, 0)].reshape([x.shape[0], x.shape[2], x.shape[1] * x.shape[3]])
    }

    let k =
      moveHeadsToOuter(kProj(x, weightGradBackend: config.WeightGradBackend))
      / sqrt(sqrt(Float(config.HeadDim)))
    let v = moveHeadsToOuter(vProj(x, weightGradBackend: config.WeightGradBackend))
    let q =
      moveHeadsToOuter(qProj(x, weightGradBackend: config.WeightGradBackend))
      / sqrt(sqrt(Float(config.HeadDim)))

    let energy = Tensor.batchedMatmul(
      a: rope(q), transA: false, b: rope(k),
      transB: true, transOut: false)
    let probs = (energy + causalMask).softmax()
    let reducedValues = Tensor.batchedMatmul(
      a: probs, transA: false, b: v, transB: false, transOut: false)
    return outProj(moveHeadsToInner(reducedValues), weightGradBackend: config.WeightGradBackend)
  }
}

class Block: Trainable {
  let config: TransformerConfig

  @Child var attn: Attention
  @Child var norm1: LayerNorm
  @Child var norm2: LayerNorm
  @Child var lin1: Linear
  @Child var lin2: Linear

  init(config: TransformerConfig) {
    self.config = config
    super.init()
    self.attn = Attention(config: config)
    self.norm1 = LayerNorm(shape: [config.ModelDim])
    self.norm2 = LayerNorm(shape: [config.ModelDim])
    self.lin1 = Linear(inCount: config.ModelDim, outCount: config.ModelDim * 2, bias: false)
    self.lin2 = Linear(inCount: config.ModelDim * 2, outCount: config.ModelDim, bias: false)
  }

  func callAsFunction(_ x: Tensor) -> Tensor {
    var h = x
    h = h + attn(norm1(h))
    h =
      h
      + lin2(
        lin1(norm2(h), weightGradBackend: config.WeightGradBackend).gelu(),
        weightGradBackend: config.WeightGradBackend)
    return h
  }
}

class Transformer: Trainable {
  let config: TransformerConfig

  @Param var embed: Tensor
  @Child var layers: TrainableArray<Block>
  @Child var normOut: LayerNorm
  @Child var unembed: Linear

  init(config: TransformerConfig) {
    self.config = config
    super.init()
    embed = Tensor(randn: [config.VocabSize, config.ModelDim])
    layers = TrainableArray((0..<config.LayerCount).map { _ in Block(config: config) })
    normOut = LayerNorm(shape: [config.ModelDim])

    unembed = Linear(
      inCount: config.ModelDim, outCount: config.VocabSize, bias: false)

    // Uniform initial probability
    unembed.weight = unembed.weight.noGrad() * 0
  }

  func callAsFunction(_ x: Tensor) -> Tensor {
    // Input should be a [N x T] tensor of indices
    var h = embed.gather(axis: 0, indices: x.flatten()).reshape([
      x.shape[0], x.shape[1], config.ModelDim,
    ])

    for layer in layers.children {
      h = layer(h)
    }
    h = normOut(h)
    h = unembed(h, weightGradBackend: config.WeightGradBackend)
    return h
  }

  func paramNorm() async throws -> Float {
    try await parameters.map { (_, param) in param.data!.pow(2).sum() }
      .reduce(
        Tensor(zeros: []), { $0 + $1 }
      ).sqrt().item()
  }
}
