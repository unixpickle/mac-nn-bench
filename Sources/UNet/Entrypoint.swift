import Foundation
import Honeycrisp

@main
struct Main {
  static func main() async {
    do {
      let flopCounter = BackendFLOPCounter(
        wrapping: try MPSBackend(allocator: .heap(10_000_000_000)))
      Backend.defaultBackend = flopCounter

      let config = UNet.Config()
      let model = UNet(config: config)
      let batch = Tensor(zeros: [32, 3, 64, 64])
      let opt = Adam(model.parameters, lr: 0.001)

      func computeLoss(_ inputs: Tensor) -> Tensor {
        let output = model(inputs)
        return output.pow(2).mean()
      }

      func step(_ inputs: Tensor) {
        let loss = computeLoss(inputs)
        loss.backward()
        opt.step()
        opt.clearGrads()
      }

      func waitForModel() async throws {
        for (_, param) in model.parameters {
          try await param.data!.wait()
        }
      }

      print("warming up...")
      var numFlops: Double = 0
      for i in 0..<5 {
        step(batch)
        if i == 0 {
          try await waitForModel()
          numFlops = Double(flopCounter.flopCount)
          print(" - estimated FLOPs in fwd+bwd: \(numFlops)")
        }
      }
      try await waitForModel()

      print("benchmarking...")
      let numIts = 20
      let t1 = DispatchTime.now()
      for _ in 0..<numIts {
        step(batch)
      }
      try await waitForModel()
      let elapsed =
        Double(DispatchTime.now().uptimeNanoseconds - t1.uptimeNanoseconds) / Double(numIts)

      let gflops = numFlops / elapsed
      print("gflops: \(gflops)")
    } catch {
      print("FATAL ERROR: \(error)")
    }
  }
}
