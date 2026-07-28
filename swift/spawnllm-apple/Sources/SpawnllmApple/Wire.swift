import Foundation

public enum UseCase: String, Decodable, Sendable {
    case general
    case contentTagging = "content_tagging"
}

public enum Guardrails: String, Decodable, Sendable {
    case `default`
    case permissiveContentTransformations = "permissive_content_transformations"
}

public enum Sampling: Decodable, Equatable, Sendable {
    case greedy
    case random(top: Int?, probabilityThreshold: Double?, seed: UInt64?)

    private enum CodingKeys: String, CodingKey {
        case mode
        case top
        case probabilityThreshold = "probability_threshold"
        case seed
    }

    private enum Mode: String, Decodable {
        case greedy
        case random
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        switch try container.decode(Mode.self, forKey: .mode) {
        case .greedy:
            self = .greedy
        case .random:
            self = .random(
                top: try container.decodeIfPresent(Int.self, forKey: .top),
                probabilityThreshold: try container.decodeIfPresent(Double.self, forKey: .probabilityThreshold),
                seed: try container.decodeIfPresent(UInt64.self, forKey: .seed)
            )
        }
    }
}

public struct Options: Decodable, Equatable, Sendable {
    public let temperature: Double?
    public let maximumResponseTokens: Int?
    public let sampling: Sampling?

    private enum CodingKeys: String, CodingKey {
        case temperature
        case maximumResponseTokens = "maximum_response_tokens"
        case sampling
    }
}

public struct Request: Decodable, Sendable {
    public let prompt: String
    public let instructions: String?
    public let useCase: UseCase
    public let guardrails: Guardrails
    public let options: Options?
    public let schema: JSONValue?

    private enum CodingKeys: String, CodingKey {
        case prompt
        case instructions
        case useCase = "use_case"
        case guardrails
        case options
        case schema
    }

    public static func decode(_ data: Data) throws -> Request {
        guard let request = try? JSONDecoder().decode(Request.self, from: data) else {
            throw SidecarError.malformedRequest
        }
        return request
    }
}

public enum Reply: Encodable, Equatable, Sendable {
    case ok(text: String)
    case failure(kind: ErrorKind, message: String)

    private enum CodingKeys: String, CodingKey {
        case status
        case text
        case kind
        case message
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .ok(let text):
            try container.encode("ok", forKey: .status)
            try container.encode(text, forKey: .text)
        case .failure(let kind, let message):
            try container.encode("error", forKey: .status)
            try container.encode(kind.rawValue, forKey: .kind)
            try container.encode(message, forKey: .message)
        }
    }

    public func line() -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.withoutEscapingSlashes]
        return try! encoder.encode(self) + Data("\n".utf8)
    }
}

public struct Probe: Encodable, Equatable, Sendable {
    public let available: Bool
    public let reason: String?

    private enum CodingKeys: String, CodingKey {
        case available
        case reason
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(available, forKey: .available)
        try container.encode(reason, forKey: .reason)
    }

    public func line() -> Data {
        try! JSONEncoder().encode(self) + Data("\n".utf8)
    }
}
