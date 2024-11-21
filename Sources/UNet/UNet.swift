import HCBacktrace
import Honeycrisp

public class Upsample: Trainable {
  @Child(name: "conv") public var conv: Conv2D

  public init(channels: Int) {
    super.init()
    self.conv = Conv2D(
      inChannels: channels, outChannels: channels, kernelSize: .square(3), padding: .same)
  }

  @recordCaller
  private func _callAsFunction(_ x: Tensor) -> Tensor {
    let upsampled = x.unsqueeze(axis: -2).unsqueeze(axis: -1).repeating(axis: -1, count: 2)
      .repeating(axis: -3, count: 2).flatten(startAxis: -2, endAxis: -1).flatten(
        startAxis: -3, endAxis: -2)
    return conv(upsampled)
  }
}

public class Downsample: Trainable {
  @Child(name: "conv") public var conv: Conv2D

  public init(channels: Int) {
    super.init()
    self.conv = Conv2D(
      inChannels: channels, outChannels: channels, kernelSize: .square(3), stride: .square(2),
      padding: .allSides(1))
  }

  @recordCaller
  private func _callAsFunction(_ x: Tensor) -> Tensor {
    conv(x)
  }
}

public class ResBlock: Trainable {
  public enum Resample {
    case none
    case upsample
    case downsample
  }

  let inChannels: Int
  let outChannels: Int
  let resample: Resample

  @Child(name: "downsample") public var downsample: Downsample
  @Child(name: "downsampleSkip") public var downsampleSkip: Downsample
  @Child(name: "upsample") public var upsample: Upsample
  @Child(name: "upsampleSkip") public var upsampleSkip: Upsample
  @Child(name: "skipConv") public var skipConv: Conv2D
  @Child(name: "inputNorm") public var inputNorm: GroupNorm
  @Child(name: "inputConv") public var inputConv: Conv2D
  @Child(name: "outputNorm") public var outputNorm: GroupNorm
  @Child(name: "outputConv") public var outputConv: Conv2D

  public init(inChannels: Int, outChannels: Int? = nil, resample: Resample = .none) {
    self.inChannels = inChannels
    self.outChannels = outChannels ?? inChannels
    self.resample = resample
    super.init()
    let outChannels = outChannels ?? inChannels
    switch resample {
    case .none:
      break
    case .upsample:
      upsample = Upsample(channels: inChannels)
      upsampleSkip = Upsample(channels: inChannels)
    case .downsample:
      downsample = Downsample(channels: inChannels)
      downsampleSkip = Downsample(channels: inChannels)
    }
    if outChannels != inChannels {
      skipConv = Conv2D(
        inChannels: inChannels, outChannels: outChannels, kernelSize: .square(3), padding: .same)
    }
    inputNorm = GroupNorm(groupCount: 32, channelCount: inChannels)
    inputConv = Conv2D(
      inChannels: inChannels, outChannels: outChannels, kernelSize: .square(3), padding: .same)
    outputNorm = GroupNorm(groupCount: 32, channelCount: outChannels)
    outputConv = Conv2D(
      inChannels: outChannels, outChannels: outChannels, kernelSize: .square(3), padding: .same)
  }

  @recordCaller
  private func _callAsFunction(_ x: Tensor) -> Tensor {
    var x = x
    var h = inputNorm(x)
    switch resample {
    case .none:
      break
    case .upsample:
      h = upsample(h)
      x = upsampleSkip(x)
    case .downsample:
      h = downsample(h)
      x = downsampleSkip(x)
    }
    h = h.silu()
    h = inputConv(h)

    h = outputNorm(h)
    h = h.silu()
    h = outputConv(h)
    if inChannels != outChannels {
      x = skipConv(x)
    }
    alwaysAssert(x.shape == h.shape, "\(x.shape) must be equal to \(h.shape)")
    return x + h
  }
}

public class UNet: Trainable {

  public struct Config: Codable, Equatable {
    public var inChannels: Int = 3
    public var outChannels: Int = 3
    public var resBlockCount: Int = 2
    public var innerChannels: [Int] = [32, 64, 64, 128]

    public init(
      inChannels: Int = 3,
      outChannels: Int = 3,
      resBlockCount: Int = 2,
      innerChannels: [Int] = [32, 64, 64, 128]
    ) {
      self.inChannels = inChannels
      self.outChannels = outChannels
      self.resBlockCount = resBlockCount
      self.innerChannels = innerChannels
    }
  }

  public class OutputBlock: Trainable {
    var hasUpsample: Bool = false
    @Child var input: ResBlock
    @Child var upsample: ResBlock

    public init(_ input: ResBlock, upsample: ResBlock? = nil) {
      super.init()
      self.input = input
      if upsample != nil {
        hasUpsample = true
        self.upsample = upsample!
      }
    }

    @recordCaller
    private func _callAsFunction(_ x: Tensor) -> Tensor {
      var h = input(x)
      if hasUpsample {
        h = upsample(h)
      }
      return h
    }
  }

  var config: Config
  @Child var inputConv: Conv2D
  @Child var inputBlocks: TrainableArray<ResBlock>
  @Child var middleBlocks: TrainableArray<ResBlock>
  @Child var outputBlocks: TrainableArray<OutputBlock>
  @Child var outputNorm: GroupNorm
  @Child var outputConv: Conv2D

  public init(config: Config) {
    self.config = config

    super.init()

    inputConv = Conv2D(
      inChannels: config.inChannels, outChannels: config.innerChannels[0], kernelSize: .square(3),
      padding: .same)

    var skipChannels: [Int] = [config.innerChannels[0]]
    var ch = config.innerChannels[0]

    var inputs = [ResBlock]()
    for i in 1..<config.innerChannels.count {
      let newCh = config.innerChannels[i]
      for _ in 0..<config.resBlockCount {
        inputs.append(ResBlock(inChannels: ch, outChannels: newCh))
        ch = newCh
        skipChannels.append(ch)
      }
      if i + 1 < config.innerChannels.count {
        inputs.append(
          ResBlock(inChannels: ch, outChannels: config.innerChannels[i], resample: .downsample))
        skipChannels.append(ch)
      }
    }

    var middle = [ResBlock]()
    middle.append(ResBlock(inChannels: ch))
    middle.append(ResBlock(inChannels: ch))

    var outputs = [OutputBlock]()
    for i in (1..<config.innerChannels.count).reversed() {
      let outChannels = config.innerChannels[i - 1]
      for j in 0..<(config.resBlockCount + 1) {
        let skip = skipChannels.popLast()!
        let inputBlock = ResBlock(inChannels: ch + skip, outChannels: outChannels)
        ch = outChannels
        let upsample: ResBlock? =
          if i > 1 && j == config.resBlockCount {
            ResBlock(inChannels: ch, resample: .upsample)
          } else {
            nil
          }
        outputs.append(OutputBlock(inputBlock, upsample: upsample))
      }
    }

    outputNorm = GroupNorm(groupCount: 32, channelCount: config.innerChannels[0])
    outputConv = Conv2D(
      inChannels: config.innerChannels[0], outChannels: config.outChannels, kernelSize: .square(3),
      padding: .same
    )

    inputBlocks = TrainableArray(inputs)
    middleBlocks = TrainableArray(middle)
    outputBlocks = TrainableArray(outputs)
  }

  public func addResolution(_ outerChannels: Int) {
    var config = config
    config.innerChannels.insert(outerChannels, at: 0)
    self.config = config

    inputConv = Conv2D(
      inChannels: config.inChannels, outChannels: config.innerChannels[0], kernelSize: .square(3),
      padding: .same)

    var skipChannels: [Int] = [config.innerChannels[0]]
    var ch = config.innerChannels[0]

    var inputs = [ResBlock]()
    let newCh = config.innerChannels[1]
    for _ in 0..<config.resBlockCount {
      inputs.append(ResBlock(inChannels: ch, outChannels: newCh))
      ch = newCh
      skipChannels.append(ch)
    }
    inputs.append(
      ResBlock(inChannels: ch, outChannels: config.innerChannels[1], resample: .downsample))
    inputs.append(contentsOf: inputBlocks.children)
    inputBlocks = TrainableArray(inputs)

    var outputs = outputBlocks.children
    outputs.last!.upsample = ResBlock(inChannels: config.innerChannels[1], resample: .upsample)

    let outChannels = config.innerChannels[0]
    for _ in 0..<(config.resBlockCount + 1) {
      let skip = skipChannels.popLast()!
      let inputBlock = ResBlock(inChannels: ch + skip, outChannels: outChannels)
      ch = outChannels
      outputs.append(OutputBlock(inputBlock, upsample: nil))
    }
    outputBlocks = TrainableArray(outputs)

    outputNorm = GroupNorm(groupCount: 32, channelCount: config.innerChannels[0])
    outputConv = Conv2D(
      inChannels: config.innerChannels[0], outChannels: config.outChannels, kernelSize: .square(3),
      padding: .same
    )

    inputBlocks = TrainableArray(inputs)
    outputBlocks = TrainableArray(outputs)
  }

  @recordCaller
  private func _callAsFunction(_ x: Tensor) -> Tensor {
    var h = x
    var skips = [Tensor]()

    h = inputConv(h)
    skips.append(h)
    for inBlock in inputBlocks.children {
      h = inBlock(h)
      skips.append(h)
    }

    alwaysAssert(skips.count == outputBlocks.children.count)

    for block in middleBlocks.children {
      h = block(h)
    }
    for outBlock in outputBlocks.children {
      h = Tensor(concat: [h, skips.popLast()!], axis: 1)
      h = outBlock(h)
    }
    h = outputNorm(h)
    h = h.silu()
    h = outputConv(h)
    return h
  }
}
