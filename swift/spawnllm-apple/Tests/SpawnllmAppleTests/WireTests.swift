import Foundation
import FoundationModels
import Testing

@testable import SpawnllmApple

func context() -> LanguageModelSession.GenerationError.Context {
    .init(debugDescription: "512 of 4096 tokens, rate limited, overloaded")
}

func request(nestedTo levels: Int) -> Data {
    let schema = (0..<levels).reduce(##"{"type":"string"}"##) { inner, _ in
        ##"{"type":"array","items":\##(inner)}"##
    }
    return Data(##"{"prompt":"hi","use_case":"general","guardrails":"default","schema":\##(schema)}"##.utf8)
}

@Suite("wire protocol")
struct WireTests {
    @Test("a minimal request decodes")
    func minimalRequest() throws {
        let request = try Request.decode(
            Data(##"{"prompt":"Reply with only: pong","use_case":"general","guardrails":"default"}"##.utf8)
        )

        #expect(request.prompt == "Reply with only: pong")
        #expect(request.instructions == nil)
        #expect(request.useCase == .general)
        #expect(request.guardrails == .default)
        #expect(request.options == nil)
        #expect(request.schema == nil)
    }

    @Test("a full request decodes every knob")
    func fullRequest() throws {
        let request = try Request.decode(
            Data(
                ##"""
                {"prompt":"tag it","instructions":"be terse","use_case":"content_tagging",
                 "guardrails":"permissive_content_transformations",
                 "options":{"temperature":0.2,"maximum_response_tokens":64,
                            "sampling":{"mode":"random","top":20,"probability_threshold":null,"seed":7}},
                 "schema":{"type":"object","properties":{"first_name":{"type":"string"}}}}
                """##.utf8
            )
        )

        #expect(request.instructions == "be terse")
        #expect(request.useCase == .contentTagging)
        #expect(request.guardrails == .permissiveContentTransformations)
        #expect(request.options?.temperature == 0.2)
        #expect(request.options?.maximumResponseTokens == 64)
        #expect(request.options?.sampling == .random(top: 20, probabilityThreshold: nil, seed: 7))
        #expect(request.schema?["properties"]?["first_name"] == .object(["type": .string("string")]))
    }

    @Test(
        "an unusable request fails with a structured error",
        arguments: [
            "not json at all",
            "",
            ##"{"use_case":"general","guardrails":"default"}"##,
            ##"{"prompt":"hi","use_case":"nonsense","guardrails":"default"}"##,
            ##"{"prompt":"hi","use_case":"general","guardrails":"developer_managed"}"##,
            ##"{"prompt":"hi","use_case":"general","guardrails":"default","options":{"sampling":{"mode":"random","seed":-1}}}"##,
            ##"{"prompt":"hi","use_case":"general","guardrails":"default","options":{"sampling":{"mode":"random","seed":18446744073709551616}}}"##,
        ]
    )
    func malformedRequest(source: String) {
        #expect(throws: SidecarError.malformedRequest) { try Request.decode(Data(source.utf8)) }
    }

    @Test("a request nested past the parser's container limit fails without blaming its shape")
    func nestingPastTheParserLimit() {
        #expect(throws: Never.self) { try Request.decode(request(nestedTo: 8)) }
        #expect(throws: SidecarError.malformedRequest) { try Request.decode(request(nestedTo: 600)) }
        #expect(!SidecarError.malformedRequest.message.contains("JSON object"))
    }

    @Test("a seed spans the whole unsigned range")
    func unsignedSeed() throws {
        let request = try Request.decode(
            Data(
                ##"""
                {"prompt":"hi","use_case":"general","guardrails":"default",
                 "options":{"sampling":{"mode":"random","seed":18446744073709551615}}}
                """##.utf8
            )
        )

        #expect(request.options?.sampling == .random(top: nil, probabilityThreshold: nil, seed: UInt64.max))
    }

    @Test("a number no fixed-width integer can hold reads as no integer at all")
    func integerConversion() {
        #expect(JSONValue.integer(Int.max).integerValue == Int.max)
        #expect(JSONValue.number(2.0).integerValue == 2)
        #expect(JSONValue.number(1.5).integerValue == nil)
        #expect(JSONValue.number(1e20).integerValue == nil)
        #expect(JSONValue.number(-1e20).integerValue == nil)
        #expect(JSONValue.number(.infinity).integerValue == nil)
        #expect(JSONValue.number(.nan).integerValue == nil)
    }

    @Test("a reply serializes as one line of JSON")
    func replies() throws {
        let line = Reply.ok(text: "a/b").line()
        #expect(String(decoding: line, as: UTF8.self).contains(##""a/b""##))
        #expect(
            try JSONDecoder().decode(JSONValue.self, from: line)
                == .object(["status": .string("ok"), "text": .string("a/b")])
        )

        let failure = try JSONDecoder().decode(JSONValue.self, from: SidecarError.rateLimited.reply.line())
        #expect(failure["status"]?.stringValue == "error")
        #expect(failure["kind"]?.stringValue == "RateLimitedError")
    }

    @Test(
        "every framework error resolves to its apple_fm_sdk class name",
        arguments: [
            (LanguageModelSession.GenerationError.exceededContextWindowSize(context()), "ExceededContextWindowSizeError"),
            (.assetsUnavailable(context()), "AssetsUnavailableError"),
            (.guardrailViolation(context()), "GuardrailViolationError"),
            (.unsupportedGuide(context()), "UnsupportedGuideError"),
            (.unsupportedLanguageOrLocale(context()), "UnsupportedLanguageOrLocaleError"),
            (.decodingFailure(context()), "DecodingFailureError"),
            (.rateLimited(context()), "RateLimitedError"),
            (.concurrentRequests(context()), "ConcurrentRequestsError"),
            (.refusal(.init(transcriptEntries: []), context()), "RefusalError"),
        ]
    )
    func resolvesFrameworkErrors(error: LanguageModelSession.GenerationError, kind: String) {
        #expect(SidecarError.resolve(error).kind.rawValue == kind)
    }

    @Test("an unrecognized error resolves to the framework base class name")
    func resolvesUnknown() {
        #expect(SidecarError.resolve(CocoaError(.fileNoSuchFile)) == .unknown)
        #expect(SidecarError.unknown.kind.rawValue == "FoundationModelsError")
    }

    @Test(
        "no message carries a digit a transient-response matcher would fire on",
        arguments: [
            SidecarError.malformedRequest, .conflictingSampling, .schemaNotObject, .unsupportedConstruct,
            .unsupportedReference, .invertedBounds, .invalidSchema, .deviceNotEligible,
            .appleIntelligenceNotEnabled, .modelNotReady, .exceededContextWindowSize, .assetsUnavailable,
            .guardrailViolation, .unsupportedGuide, .unsupportedLanguageOrLocale, .decodingFailure,
            .rateLimited, .concurrentRequests, .refusal, .generation, .unknown,
        ]
    )
    func messagesCarryNoDigits(error: SidecarError) {
        #expect(error.message.allSatisfy { !$0.isNumber })
    }

    @Test("a framework error's debug description never reaches the message")
    func messagesDropDebugDescriptions() {
        #expect(SidecarError.resolve(LanguageModelSession.GenerationError.rateLimited(context())) == .rateLimited)
        #expect(!SidecarError.rateLimited.message.contains("4096"))
    }

    @Test("sampling maps onto the framework's factories")
    func sampling() throws {
        #expect(try Sidecar.samplingMode(nil) == nil)
        #expect(try Sidecar.samplingMode(.greedy) == .greedy)
        #expect(
            try Sidecar.samplingMode(.random(top: 20, probabilityThreshold: nil, seed: 7)) == .random(top: 20, seed: 7)
        )
        #expect(
            try Sidecar.samplingMode(.random(top: nil, probabilityThreshold: 0.9, seed: nil))
                == .random(probabilityThreshold: 0.9, seed: nil)
        )
        #expect(
            try Sidecar.samplingMode(.random(top: nil, probabilityThreshold: nil, seed: 3))
                == .random(probabilityThreshold: 1.0, seed: 3)
        )
        #expect(throws: SidecarError.conflictingSampling) {
            try Sidecar.samplingMode(.random(top: 20, probabilityThreshold: 0.9, seed: nil))
        }
    }

    @Test("options default to the framework's own defaults when unset")
    func options() throws {
        #expect(try Sidecar.generationOptions(nil) == GenerationOptions())
        #expect(
            try Sidecar.generationOptions(
                JSONDecoder().decode(Options.self, from: Data(##"{"temperature":0.5}"##.utf8))
            ) == GenerationOptions(temperature: 0.5)
        )
    }

    @Test("a use case maps onto the framework's own")
    func useCases() {
        #expect(Sidecar.useCase(.general) == .general)
        #expect(Sidecar.useCase(.contentTagging) == .contentTagging)
    }
}
