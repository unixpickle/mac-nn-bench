// swift-tools-version: 5.10
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
  name: "MacNNBench",
  platforms: [
    .macOS(.v13)
  ],
  products: [],
  dependencies: [
    .package(url: "https://github.com/unixpickle/honeycrisp.git", from: "0.0.4")
  ],
  targets: [
    .executableTarget(
      name: "Transformer",
      dependencies: [
        .product(name: "Honeycrisp", package: "honeycrisp")
      ]
    ),
    .executableTarget(
      name: "TransformerANE",
      dependencies: [
        .product(name: "Honeycrisp", package: "honeycrisp")
      ]
    ),
  ]
)
