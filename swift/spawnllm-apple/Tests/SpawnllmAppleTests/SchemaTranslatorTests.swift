import Foundation
import FoundationModels
import Testing

@testable import SpawnllmApple

func translated(_ source: String) throws -> JSONValue {
    let schema = try SchemaTranslator.generationSchema(for: JSONDecoder().decode(JSONValue.self, from: Data(source.utf8)))
    return try JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode(schema))
}

func nested(_ template: String, levels: Int) -> String {
    let inner = (0..<levels).reduce(##"{"type":"string"}"##) { inner, _ in
        template.replacingOccurrences(of: "INNER", with: inner)
    }
    return ##"{"type":"object","title":"Deep","required":["deep"],"properties":{"deep":\##(inner)}}"##
}

@Suite("schema translation")
struct SchemaTranslatorTests {
    @Test("a pattern reaches the schema desugared")
    func pattern() throws {
        let schema = try translated(
            ##"""
            {"type":"object","title":"Sku","x-order":["code","quantity"],"required":["code","quantity"],
             "properties":{"code":{"type":"string","pattern":"^[A-Z]{3}-\\d{4}$","description":"the sku"},
                           "quantity":{"type":"integer","minimum":1,"maximum":100}}}
            """##
        )

        #expect(schema["properties"]?["code"]?["pattern"]?.stringValue == #"\w{3}\-\d{4}"#)
        #expect(schema["properties"]?["code"]?["description"]?.stringValue == "the sku")
        #expect(schema["properties"]?["quantity"]?["minimum"]?.integerValue == 1)
        #expect(schema["properties"]?["quantity"]?["maximum"]?.integerValue == 100)
        #expect(schema["required"] == .array([.string("code"), .string("quantity")]))
    }

    @Test("a pattern outside the accepted subset is dropped, not fatal")
    func unsafePattern() throws {
        let schema = try translated(
            ##"{"type":"object","title":"Bad","required":["x"],"properties":{"x":{"type":"string","pattern":"[^a]"}}}"##
        )

        #expect(schema["properties"]?["x"] == .object(["type": .string("string")]))
    }

    @Test("a property absent from required is optional")
    func optionalProperty() throws {
        let schema = try translated(
            ##"""
            {"type":"object","title":"Order","x-order":["id","note"],"required":["id"],
             "properties":{"id":{"type":"string"},"note":{"type":"string"}}}
            """##
        )

        #expect(schema["required"] == .array([.string("id")]))
        #expect(schema["x-order"] == .array([.string("id"), .string("note")]))
    }

    @Test("x-order fixes property order")
    func declaredOrder() throws {
        let schema = try translated(
            ##"""
            {"type":"object","title":"Ordered","x-order":["zeta","alpha"],"required":[],
             "properties":{"alpha":{"type":"integer"},"zeta":{"type":"integer"}}}
            """##
        )

        #expect(schema["x-order"] == .array([.string("zeta"), .string("alpha")]))
    }

    @Test("a nested object keeps its title and its own properties")
    func nestedObject() throws {
        let schema = try translated(
            ##"""
            {"type":"object","title":"Root","required":["home"],
             "properties":{"home":{"type":"object","title":"Address","required":["city","zip"],
                                   "x-order":["city","zip"],
                                   "properties":{"city":{"type":"string"},
                                                 "zip":{"type":"string","pattern":"^[0-9]{5}$"}}}}}
            """##
        )

        #expect(schema["properties"]?["home"]?["$ref"]?.stringValue == "#/$defs/Address")
        #expect(schema["$defs"]?["Address"]?["x-order"] == .array([.string("city"), .string("zip")]))
        #expect(schema["$defs"]?["Address"]?["properties"]?["zip"]?["pattern"]?.stringValue == "\\d{5}")
    }

    @Test("an array carries its item schema and its element bounds")
    func arrayBounds() throws {
        let schema = try translated(
            ##"""
            {"type":"object","title":"Cart","required":["lines"],
             "properties":{"lines":{"type":"array","minItems":1,"maxItems":3,
                                    "items":{"type":"object","title":"Line","required":["sku"],
                                             "properties":{"sku":{"type":"string"}}}}}}
            """##
        )

        #expect(schema["properties"]?["lines"]?["minItems"]?.integerValue == 1)
        #expect(schema["properties"]?["lines"]?["maxItems"]?.integerValue == 3)
        #expect(schema["properties"]?["lines"]?["items"]?["$ref"]?.stringValue == "#/$defs/Line")
        #expect(schema["$defs"]?["Line"]?["properties"]?["sku"] == .object(["type": .string("string")]))
    }

    @Test("a string enum becomes a choice of constants")
    func stringEnum() throws {
        let schema = try translated(
            ##"""
            {"type":"object","title":"Pick","required":["color"],
             "properties":{"color":{"type":"string","enum":["red","green"]}}}
            """##
        )

        let choices = try #require(schema["$defs"]?["PickColor"]?["anyOf"]?.arrayValue)
        #expect(choices.compactMap { $0["enum"]?.arrayValue?.first?.stringValue } == ["red", "green"])
    }

    @Test("a const-only anyOf becomes a choice of constants")
    func constUnion() throws {
        let schema = try translated(
            ##"""
            {"type":"object","title":"Pick","required":["size"],
             "properties":{"size":{"anyOf":[{"const":"small"},{"const":"large"}]}}}
            """##
        )

        let choices = try #require(schema["$defs"]?["PickSize"]?["anyOf"]?.arrayValue)
        #expect(choices.compactMap { $0["enum"]?.arrayValue?.first?.stringValue } == ["small", "large"])
    }

    @Test("a $ref resolves against $defs")
    func reference() throws {
        let schema = try translated(
            ##"""
            {"type":"object","title":"Root","required":["home"],
             "properties":{"home":{"$ref":"#/$defs/Address","description":"where they live"}},
             "$defs":{"Address":{"type":"object","title":"Address","required":["city"],
                                 "properties":{"city":{"type":"string"}}}}}
            """##
        )

        #expect(schema["properties"]?["home"]?["$ref"]?.stringValue == "#/$defs/Address")
        #expect(schema["properties"]?["home"]?["description"]?.stringValue == "where they live")
        #expect(schema["$defs"]?["Address"]?["properties"]?["city"] == .object(["type": .string("string")]))
    }

    @Test("numeric bounds reach the schema, exclusive integer bounds tightened by one")
    func bounds() throws {
        let schema = try translated(
            ##"""
            {"type":"object","title":"Ranges","required":["count","score"],
             "properties":{"count":{"type":"integer","exclusiveMinimum":0,"exclusiveMaximum":10},
                           "score":{"type":"number","minimum":0.0,"maximum":1.0}}}
            """##
        )

        #expect(schema["properties"]?["count"]?["minimum"]?.integerValue == 1)
        #expect(schema["properties"]?["count"]?["maximum"]?.integerValue == 9)
        #expect(schema["properties"]?["score"]?["minimum"]?.doubleValue == 0.0)
        #expect(schema["properties"]?["score"]?["maximum"]?.doubleValue == 1.0)
    }

    @Test("a nullable type collapses to its non-null member")
    func nullableType() throws {
        let schema = try translated(
            ##"""
            {"type":"object","title":"Maybe","required":[],
             "properties":{"note":{"type":["string","null"]}}}
            """##
        )

        #expect(schema["properties"]?["note"] == .object(["type": .string("string")]))
    }

    @Test(
        "a nullable union collapses to its non-null member",
        arguments: [
            ##"{"anyOf":[{"type":"string"},{"type":"null"}],"default":null,"title":"Email"}"##,
            ##"{"anyOf":[{"type":"null"},{"type":"string"}]}"##,
            ##"{"anyOf":[{"type":"string"}]}"##,
            ##"{"oneOf":[{"type":"string"},{"type":"null"}]}"##,
        ]
    )
    func nullableUnion(property: String) throws {
        let schema = try translated(
            ##"""
            {"type":"object","title":"Contact","x-order":["name","email"],"required":["name"],
             "properties":{"name":{"type":"string"},"email":\##(property)}}
            """##
        )

        #expect(schema["properties"]?["email"] == .object(["type": .string("string")]))
        #expect(schema["required"] == .array([.string("name")]))
        #expect(schema["x-order"] == .array([.string("name"), .string("email")]))
    }

    @Test("a nullable member listed in required stays required")
    func requiredNullable() throws {
        let schema = try translated(
            ##"""
            {"type":"object","title":"Ticket","required":["email"],
             "properties":{"email":{"anyOf":[{"type":"string"},{"type":"null"}],"default":null}}}
            """##
        )

        #expect(schema["properties"]?["email"] == .object(["type": .string("string")]))
        #expect(schema["required"] == .array([.string("email")]))
    }

    @Test("a nullable enum keeps its choices")
    func nullableEnum() throws {
        let schema = try translated(
            ##"""
            {"type":"object","title":"Ticket","required":[],
             "properties":{"status":{"anyOf":[{"type":"string","enum":["open","closed"]},{"type":"null"}]},
                           "size":{"anyOf":[{"const":"small"},{"const":"large"},{"type":"null"}]}}}
            """##
        )

        let states = try #require(schema["$defs"]?["TicketStatus"]?["anyOf"]?.arrayValue)
        #expect(states.compactMap { $0["enum"]?.arrayValue?.first?.stringValue } == ["open", "closed"])
        let sizes = try #require(schema["$defs"]?["TicketSize"]?["anyOf"]?.arrayValue)
        #expect(sizes.compactMap { $0["enum"]?.arrayValue?.first?.stringValue } == ["small", "large"])
    }

    @Test("a nullable object and a nullable array keep their own shape")
    func nullableComposites() throws {
        let schema = try translated(
            ##"""
            {"type":"object","title":"Root","required":[],
             "properties":{"home":{"anyOf":[{"$ref":"#/$defs/Address"},{"type":"null"}]},
                           "tags":{"anyOf":[{"type":"array","items":{"type":"string"},"maxItems":3},
                                            {"type":"null"}]}},
             "$defs":{"Address":{"type":"object","title":"Address","required":["city"],
                                 "properties":{"city":{"type":"string"}}}}}
            """##
        )

        #expect(schema["properties"]?["home"]?["$ref"]?.stringValue == "#/$defs/Address")
        #expect(schema["$defs"]?["Address"]?["properties"]?["city"] == .object(["type": .string("string")]))
        #expect(schema["properties"]?["tags"]?["maxItems"]?.integerValue == 3)
        #expect(schema["properties"]?["tags"]?["items"] == .object(["type": .string("string")]))
    }

    @Test(
        "nesting past the depth cap is rejected on every recursive edge",
        arguments: [
            ##"{"type":"array","items":INNER}"##,
            ##"{"type":"object","required":["k"],"properties":{"k":INNER}}"##,
            ##"{"anyOf":[INNER,{"type":"null"}]}"##,
            ##"{"anyOf":[INNER,{"type":"integer"}]}"##,
        ]
    )
    func depthCap(template: String) throws {
        #expect(throws: Never.self) { try translated(nested(template, levels: 8)) }
        #expect(throws: SidecarError.unsupportedConstruct) {
            try translated(nested(template, levels: SchemaTranslator.maximumDepth + 1))
        }
    }

    @Test(
        "an unusable schema fails with a structured error",
        arguments: [
            (##"[]"##, SidecarError.schemaNotObject),
            (##"{"type":"object","properties":{"x":{"type":"null"}}}"##, .unsupportedConstruct),
            (##"{"type":"object","properties":{"x":{"anyOf":[]}}}"##, .unsupportedConstruct),
            (##"{"type":"object","properties":{"x":{"anyOf":[{"type":"null"}]}}}"##, .unsupportedConstruct),
            (##"{"type":"object","properties":{"x":{"type":"integer","minimum":9,"maximum":1}}}"##, .invertedBounds),
            (##"{"type":"object","properties":{"x":{"type":"integer","minimum":1e20}}}"##, .unsupportedConstruct),
            (##"{"type":"object","properties":{"x":{"type":"integer","maximum":-1e20}}}"##, .unsupportedConstruct),
            (##"{"type":"object","properties":{"x":{"type":"integer","minimum":1.5}}}"##, .unsupportedConstruct),
            (
                ##"{"type":"object","properties":{"x":{"type":"integer","exclusiveMinimum":9223372036854775807}}}"##,
                .unsupportedConstruct
            ),
            (
                ##"{"type":"object","properties":{"x":{"type":"integer","exclusiveMaximum":-9223372036854775808}}}"##,
                .unsupportedConstruct
            ),
            (
                ##"{"type":"object","properties":{"x":{"type":"array","items":{"type":"string"},"maxItems":1e20}}}"##,
                .unsupportedConstruct
            ),
            (##"{"type":"object","properties":{"x":{"type":"number","minimum":"low"}}}"##, .unsupportedConstruct),
            (##"{"type":"object","properties":{"x":{"type":"string","enum":[1,2]}}}"##, .unsupportedConstruct),
            (##"{"type":"object","properties":{"x":{"type":"array"}}}"##, .unsupportedConstruct),
            (##"{"type":"integer","enum":[1,2]}"##, .unsupportedConstruct),
            (##"{"type":"object","properties":{"x":{"$ref":"https://example.com/x"}}}"##, .unsupportedReference),
            (##"{"type":"object","properties":{"x":{"$ref":"#/$defs/Missing"}}}"##, .invalidSchema),
        ]
    )
    func rejects(source: String, expected: SidecarError) throws {
        #expect(throws: expected) { try translated(source) }
    }
}
