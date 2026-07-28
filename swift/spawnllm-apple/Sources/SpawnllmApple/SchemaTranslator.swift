import FoundationModels

final class Namer {
    private var taken: Set<String> = []

    func reserve(_ name: String) {
        taken.insert(name)
    }

    func unique(_ base: String) -> String {
        let root = base.isEmpty ? "Schema" : base
        let name =
            taken.contains(root)
            ? (2...).lazy.map { "\(root)\($0)" }.first { !self.taken.contains($0) }!
            : root
        taken.insert(name)
        return name
    }
}

/// Exclusive numeric bounds are exact on integers but inclusive on numbers, since no guide
/// expresses exclusivity. A union's `null` members drop: optionality comes from a property's
/// absence from `required`, not from a null type.
public enum SchemaTranslator {
    static let defsPrefix = "#/$defs/"
    static let definitionsPrefix = "#/definitions/"
    static let maximumDepth = 32

    public static func generationSchema(for json: JSONValue) throws -> GenerationSchema {
        guard let root = json.objectValue else { throw SidecarError.schemaNotObject }
        let namer = Namer()
        let definitions = (root["$defs"] ?? root["definitions"])?.objectValue ?? [:]
        definitions.keys.forEach(namer.reserve)
        let dependencies = try definitions.keys.sorted().map { key in
            try schema(definitions[key]!, named: key, namer: namer, depth: 0)
        }
        do {
            return try GenerationSchema(
                root: try schema(json, hint: "Schema", namer: namer, depth: 0),
                dependencies: dependencies
            )
        } catch let error as GenerationSchema.SchemaError {
            throw SidecarError.resolve(error)
        }
    }

    static func schema(_ node: JSONValue, hint: String, namer: Namer, depth: Int) throws -> DynamicGenerationSchema {
        try schema(node, named: namer.unique(node["title"]?.stringValue ?? hint), namer: namer, depth: depth)
    }

    static func schema(_ node: JSONValue, named name: String, namer: Namer, depth: Int) throws -> DynamicGenerationSchema {
        guard depth <= maximumDepth else { throw SidecarError.unsupportedConstruct }
        guard let entries = node.objectValue else { throw SidecarError.schemaNotObject }
        let description = entries["description"]?.stringValue
        if let reference = entries["$ref"]?.stringValue {
            return .init(referenceTo: try referenceName(reference))
        }
        if let choices = entries["enum"]?.arrayValue {
            return .init(name: name, description: description, anyOf: try strings(choices))
        }
        if let variants = entries["anyOf"]?.arrayValue ?? entries["oneOf"]?.arrayValue {
            return try union(variants, named: name, description: description, namer: namer, depth: depth)
        }
        switch try kind(of: entries) {
        case "object":
            return try object(entries, named: name, description: description, namer: namer, depth: depth)
        case "array":
            guard let items = entries["items"] else { throw SidecarError.unsupportedConstruct }
            return .init(
                arrayOf: try schema(items, hint: "\(name)Item", namer: namer, depth: depth + 1),
                minimumElements: try integer(entries, "minItems"),
                maximumElements: try integer(entries, "maxItems")
            )
        case "string":
            return .init(type: String.self, guides: stringGuides(entries))
        case "integer":
            return .init(type: Int.self, guides: try integerGuides(entries))
        case "number":
            return .init(type: Double.self, guides: try numberGuides(entries))
        case "boolean":
            return .init(type: Bool.self)
        default:
            throw SidecarError.unsupportedConstruct
        }
    }

    static func kind(of entries: [String: JSONValue]) throws -> String {
        switch entries["type"] {
        case .string(let kind):
            return kind
        case .array(let kinds):
            let named = try strings(kinds).filter { $0 != "null" }
            guard named.count == 1 else { throw SidecarError.unsupportedConstruct }
            return named[0]
        case nil where entries["properties"] != nil:
            return "object"
        case nil where entries["const"]?.stringValue != nil:
            return "string"
        default:
            throw SidecarError.unsupportedConstruct
        }
    }

    static func object(
        _ entries: [String: JSONValue],
        named name: String,
        description: String?,
        namer: Namer,
        depth: Int
    ) throws -> DynamicGenerationSchema {
        let properties = entries["properties"]?.objectValue ?? [:]
        let required = Set(try strings(entries["required"]?.arrayValue ?? []))
        return .init(
            name: name,
            description: description,
            properties: try order(of: properties, entries: entries).map { key in
                .init(
                    name: key,
                    description: properties[key]?["description"]?.stringValue,
                    schema: try schema(properties[key]!, hint: name + titled(key), namer: namer, depth: depth + 1),
                    isOptional: !required.contains(key)
                )
            }
        )
    }

    static func order(of properties: [String: JSONValue], entries: [String: JSONValue]) throws -> [String] {
        let declared = try strings(entries["x-order"]?.arrayValue ?? []).filter { properties[$0] != nil }
        return declared + properties.keys.filter { !declared.contains($0) }.sorted()
    }

    static func union(
        _ variants: [JSONValue],
        named name: String,
        description: String?,
        namer: Namer,
        depth: Int
    ) throws -> DynamicGenerationSchema {
        let members = variants.filter { $0["type"]?.stringValue != "null" }
        guard members.count > 1 else {
            guard let sole = members.first else { throw SidecarError.unsupportedConstruct }
            return try schema(sole, named: name, namer: namer, depth: depth + 1)
        }
        let constants = members.compactMap { $0["const"]?.stringValue }
        if constants.count == members.count {
            return .init(name: name, description: description, anyOf: constants)
        }
        return .init(
            name: name,
            description: description,
            anyOf: try members.enumerated().map { index, variant in
                try schema(variant, hint: "\(name)Choice\(index + 1)", namer: namer, depth: depth + 1)
            }
        )
    }

    static func stringGuides(_ entries: [String: JSONValue]) -> [GenerationGuide<String>] {
        let constant = entries["const"]?.stringValue.map { GenerationGuide.constant($0) }
        let pattern = entries["pattern"]?.stringValue
            .flatMap(RegexPattern.desugared)
            .flatMap { try? Regex($0) }
            .map { GenerationGuide.pattern($0) }
        return [constant, pattern].compactMap { $0 }
    }

    static func integerGuides(_ entries: [String: JSONValue]) throws -> [GenerationGuide<Int>] {
        let lower = try integer(entries, "minimum") ?? integer(entries, "exclusiveMinimum", shiftedBy: 1)
        let upper = try integer(entries, "maximum") ?? integer(entries, "exclusiveMaximum", shiftedBy: -1)
        switch (lower, upper) {
        case (let lower?, let upper?):
            guard lower <= upper else { throw SidecarError.invertedBounds }
            return [.range(lower...upper)]
        case (let lower?, nil):
            return [.minimum(lower)]
        case (nil, let upper?):
            return [.maximum(upper)]
        case (nil, nil):
            return []
        }
    }

    static func numberGuides(_ entries: [String: JSONValue]) throws -> [GenerationGuide<Double>] {
        let lower = try number(entries, "minimum") ?? number(entries, "exclusiveMinimum")
        let upper = try number(entries, "maximum") ?? number(entries, "exclusiveMaximum")
        switch (lower, upper) {
        case (let lower?, let upper?):
            guard lower <= upper else { throw SidecarError.invertedBounds }
            return [.range(lower...upper)]
        case (let lower?, nil):
            return [.minimum(lower)]
        case (nil, let upper?):
            return [.maximum(upper)]
        case (nil, nil):
            return []
        }
    }

    static func integer(_ entries: [String: JSONValue], _ key: String, shiftedBy offset: Int = 0) throws -> Int? {
        guard let bound = entries[key] else { return nil }
        guard let value = bound.integerValue, case (let shifted, false) = value.addingReportingOverflow(offset) else {
            throw SidecarError.unsupportedConstruct
        }
        return shifted
    }

    static func number(_ entries: [String: JSONValue], _ key: String) throws -> Double? {
        guard let bound = entries[key] else { return nil }
        guard let value = bound.doubleValue else { throw SidecarError.unsupportedConstruct }
        return value
    }

    static func referenceName(_ reference: String) throws -> String {
        for prefix in [defsPrefix, definitionsPrefix] where reference.hasPrefix(prefix) {
            return String(reference.dropFirst(prefix.count))
        }
        throw SidecarError.unsupportedReference
    }

    static func strings(_ values: [JSONValue]) throws -> [String] {
        let named = values.compactMap(\.stringValue)
        guard named.count == values.count else { throw SidecarError.unsupportedConstruct }
        return named
    }

    static func titled(_ key: String) -> String {
        key.split(whereSeparator: { !$0.isLetter && !$0.isNumber })
            .map { $0.prefix(1).uppercased() + $0.dropFirst() }
            .joined()
    }
}
