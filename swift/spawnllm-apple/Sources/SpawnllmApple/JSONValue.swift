import Foundation

public enum JSONValue: Sendable, Equatable {
    case null
    case bool(Bool)
    case integer(Int)
    case number(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])

    public subscript(key: String) -> JSONValue? {
        guard case .object(let entries) = self else { return nil }
        return entries[key]
    }

    public var stringValue: String? {
        guard case .string(let text) = self else { return nil }
        return text
    }

    public var integerValue: Int? {
        switch self {
        case .integer(let value): value
        case .number(let value): Int(exactly: value)
        default: nil
        }
    }

    public var doubleValue: Double? {
        switch self {
        case .integer(let value): Double(value)
        case .number(let value): value
        default: nil
        }
    }

    public var arrayValue: [JSONValue]? {
        guard case .array(let items) = self else { return nil }
        return items
    }

    public var objectValue: [String: JSONValue]? {
        guard case .object(let entries) = self else { return nil }
        return entries
    }
}

extension JSONValue: Decodable {
    public init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Int.self) {
            self = .integer(value)
        } else if let value = try? container.decode(Double.self) {
            self = .number(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([JSONValue].self) {
            self = .array(value)
        } else {
            self = .object(try container.decode([String: JSONValue].self))
        }
    }
}
