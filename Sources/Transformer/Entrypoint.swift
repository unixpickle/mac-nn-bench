import Foundation
import Honeycrisp

@main
struct Main {
  static func main() async {
    do {
      Backend.defaultBackend = try MPSBackend(allocator: .heap(12_000_000_000))

      let config = TransformerConfig(
        VocabSize: 16384 + 256, TokenCount: 128 + 256, WeightGradBackend: Backend.defaultBackend)
      let model = Transformer(config: config)
      let batch = Tensor(
        data: (0..<(4 * config.TokenCount)).map { _ in Int.random(in: 0..<config.VocabSize) },
        shape: [4, config.TokenCount])
      let opt = Adam(model.parameters, lr: 0.001)

      func computeLoss(_ inputs: Tensor) -> Tensor {
        let output = model(inputs)
        let loss = output.logSoftmax(axis: -1).gather(axis: -1, indices: inputs.unsqueeze(axis: -1))
          .mean()
        return loss
      }

      func step(_ inputs: Tensor) {
        let loss = computeLoss(inputs)
        loss.backward()
        opt.step()
        opt.clearGrads()
      }

      print("warming up...")
      for _ in 0..<5 {
        step(batch)
      }
      let _ = try await model.paramNorm()

      print("benchmarking...")
      let numIts = 20
      let t1 = DispatchTime.now()
      for _ in 0..<numIts {
        step(batch)
      }
      let _ = try await model.paramNorm()
      let elapsed =
        Double(DispatchTime.now().uptimeNanoseconds - t1.uptimeNanoseconds) / Double(numIts)

      var numParams = 0
      for (_, p) in model.parameters {
        var prod = 1
        for x in p.data!.shape {
          prod *= x
        }
        numParams += prod
      }
      let numFlops = Double(6 * batch.shape[0] * numParams * config.TokenCount)
      let gflops = numFlops / elapsed
      print("gflops: \(gflops)")
    } catch {
      print("FATAL ERROR: \(error)")
    }
  }
}
