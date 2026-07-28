import FoundationModels

public enum Sidecar {
    public static func probe() -> Probe {
        switch SystemLanguageModel.default.availability {
        case .available:
            Probe(available: true, reason: nil)
        case .unavailable(.deviceNotEligible):
            Probe(available: false, reason: "DEVICE_NOT_ELIGIBLE")
        case .unavailable(.appleIntelligenceNotEnabled):
            Probe(available: false, reason: "APPLE_INTELLIGENCE_NOT_ENABLED")
        case .unavailable(.modelNotReady):
            Probe(available: false, reason: "MODEL_NOT_READY")
        case .unavailable:
            Probe(available: false, reason: "UNKNOWN")
        }
    }

    public static func generate(_ request: Request) async -> Reply {
        do {
            return .ok(text: try await respond(request))
        } catch {
            return SidecarError.resolve(error).reply
        }
    }

    static func respond(_ request: Request) async throws -> String {
        try available()
        let session = LanguageModelSession(
            model: SystemLanguageModel(useCase: useCase(request.useCase), guardrails: guardrails(request.guardrails)),
            instructions: request.instructions
        )
        let options = try generationOptions(request.options)
        guard let schema = request.schema else {
            return try await session.respond(to: request.prompt, options: options).content
        }
        return try await session
            .respond(to: request.prompt, schema: SchemaTranslator.generationSchema(for: schema), options: options)
            .content
            .jsonString
    }

    static func available() throws {
        switch SystemLanguageModel.default.availability {
        case .available: return
        case .unavailable(.deviceNotEligible): throw SidecarError.deviceNotEligible
        case .unavailable(.appleIntelligenceNotEnabled): throw SidecarError.appleIntelligenceNotEnabled
        case .unavailable(.modelNotReady): throw SidecarError.modelNotReady
        case .unavailable: throw SidecarError.assetsUnavailable
        }
    }

    static func useCase(_ useCase: UseCase) -> SystemLanguageModel.UseCase {
        switch useCase {
        case .general: .general
        case .contentTagging: .contentTagging
        }
    }

    static func guardrails(_ guardrails: Guardrails) -> SystemLanguageModel.Guardrails {
        switch guardrails {
        case .default: .default
        case .permissiveContentTransformations: .permissiveContentTransformations
        }
    }

    static func generationOptions(_ options: Options?) throws -> GenerationOptions {
        guard let options else { return GenerationOptions() }
        return GenerationOptions(
            sampling: try samplingMode(options.sampling),
            temperature: options.temperature,
            maximumResponseTokens: options.maximumResponseTokens
        )
    }

    static func samplingMode(_ sampling: Sampling?) throws -> GenerationOptions.SamplingMode? {
        switch sampling {
        case nil:
            return nil
        case .greedy:
            return .greedy
        case .random(let top, let threshold, let seed):
            switch (top, threshold) {
            case (let top?, nil): return .random(top: top, seed: seed)
            case (nil, let threshold?): return .random(probabilityThreshold: threshold, seed: seed)
            case (nil, nil): return .random(probabilityThreshold: 1.0, seed: seed)
            case (_?, _?): throw SidecarError.conflictingSampling
            }
        }
    }
}
