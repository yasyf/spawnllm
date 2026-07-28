import FoundationModels

public enum ErrorKind: String, Sendable {
    case exceededContextWindowSize = "ExceededContextWindowSizeError"
    case assetsUnavailable = "AssetsUnavailableError"
    case guardrailViolation = "GuardrailViolationError"
    case unsupportedGuide = "UnsupportedGuideError"
    case unsupportedLanguageOrLocale = "UnsupportedLanguageOrLocaleError"
    case decodingFailure = "DecodingFailureError"
    case rateLimited = "RateLimitedError"
    case concurrentRequests = "ConcurrentRequestsError"
    case refusal = "RefusalError"
    case invalidGenerationSchema = "InvalidGenerationSchemaError"
    case generation = "GenerationError"
    case foundationModels = "FoundationModelsError"
}

/// Messages are constants and never interpolate framework text, model output, or token
/// counts: a host marks a response transient by pattern-matching the composed message, and
/// a bare three-digit number leaking through would read as a retryable 5xx.
public struct SidecarError: Error, Equatable, Sendable {
    public let kind: ErrorKind
    public let message: String

    public static let malformedRequest = SidecarError(
        kind: .foundationModels,
        message: "the request on stdin could not be decoded into a sidecar request"
    )
    public static let conflictingSampling = SidecarError(
        kind: .foundationModels,
        message: "random sampling cannot set both top and probability_threshold"
    )
    public static let schemaNotObject = SidecarError(
        kind: .invalidGenerationSchema,
        message: "the schema is not a JSON object"
    )
    public static let unsupportedConstruct = SidecarError(
        kind: .invalidGenerationSchema,
        message: "the schema uses a construct the on-device model cannot express"
    )
    public static let unsupportedReference = SidecarError(
        kind: .invalidGenerationSchema,
        message: "the schema references a definition outside its own $defs"
    )
    public static let invertedBounds = SidecarError(
        kind: .invalidGenerationSchema,
        message: "the schema declares a numeric bound whose minimum exceeds its maximum"
    )
    public static let invalidSchema = SidecarError(
        kind: .invalidGenerationSchema,
        message: "the schema does not resolve into a generation schema"
    )
    public static let deviceNotEligible = SidecarError(
        kind: .assetsUnavailable,
        message: "this device is not eligible for Apple Intelligence"
    )
    public static let appleIntelligenceNotEnabled = SidecarError(
        kind: .assetsUnavailable,
        message: "Apple Intelligence is not enabled on this device"
    )
    public static let modelNotReady = SidecarError(
        kind: .assetsUnavailable,
        message: "the on-device model is still downloading or preparing"
    )
    public static let exceededContextWindowSize = SidecarError(
        kind: .exceededContextWindowSize,
        message: "the prompt and response exceeded the model's context window"
    )
    public static let assetsUnavailable = SidecarError(
        kind: .assetsUnavailable,
        message: "the on-device model assets are unavailable"
    )
    public static let guardrailViolation = SidecarError(
        kind: .guardrailViolation,
        message: "the request tripped the on-device model's safety guardrails"
    )
    public static let unsupportedGuide = SidecarError(
        kind: .unsupportedGuide,
        message: "the schema uses a guide the on-device model does not support"
    )
    public static let unsupportedLanguageOrLocale = SidecarError(
        kind: .unsupportedLanguageOrLocale,
        message: "the prompt uses a language or locale the on-device model does not support"
    )
    public static let decodingFailure = SidecarError(
        kind: .decodingFailure,
        message: "the model's response could not be decoded into the requested schema"
    )
    public static let rateLimited = SidecarError(
        kind: .rateLimited,
        message: "the on-device model rate limited this request"
    )
    public static let concurrentRequests = SidecarError(
        kind: .concurrentRequests,
        message: "too many concurrent on-device model requests"
    )
    public static let refusal = SidecarError(
        kind: .refusal,
        message: "the on-device model refused to answer this prompt"
    )
    public static let generation = SidecarError(
        kind: .generation,
        message: "on-device generation failed"
    )
    public static let unknown = SidecarError(
        kind: .foundationModels,
        message: "the Foundation Models framework reported an error"
    )

    public static func resolve(_ error: any Error) -> SidecarError {
        switch error {
        case let sidecar as SidecarError: sidecar
        case is GenerationSchema.SchemaError: .invalidSchema
        case let generation as LanguageModelSession.GenerationError:
            switch generation {
            case .exceededContextWindowSize: .exceededContextWindowSize
            case .assetsUnavailable: .assetsUnavailable
            case .guardrailViolation: .guardrailViolation
            case .unsupportedGuide: .unsupportedGuide
            case .unsupportedLanguageOrLocale: .unsupportedLanguageOrLocale
            case .decodingFailure: .decodingFailure
            case .rateLimited: .rateLimited
            case .concurrentRequests: .concurrentRequests
            case .refusal: .refusal
            @unknown default: .generation
            }
        default: .unknown
        }
    }

    public var reply: Reply {
        .failure(kind: kind, message: message)
    }
}
