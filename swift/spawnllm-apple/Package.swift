// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "spawnllm-apple",
    platforms: [.macOS(.v26)],
    products: [
        .executable(name: "spawnllm-apple", targets: ["spawnllm-apple"])
    ],
    targets: [
        .target(name: "SpawnllmApple"),
        .executableTarget(name: "spawnllm-apple", dependencies: ["SpawnllmApple"]),
        .testTarget(name: "SpawnllmAppleTests", dependencies: ["SpawnllmApple"]),
    ]
)
