/// Foundation Models rejects bracket classes and anchors, so each class widens to the
/// narrowest accepted escape — `\d`, `\w`, or `.`. A guide therefore constrains a value's
/// shape, not its exact contents; the caller's own validation remains the exact check.
public enum RegexPattern {
    static let metacharacters: Set<Character> = [
        "\\", ".", "*", "+", "?", "(", ")", "[", "]", "{", "}", "|", "^", "$",
    ]
    static let bareEscapes: Set<Character> = ["-", "/"]
    static let classEscapes: Set<Character> = ["d", "w"]
    static let digits: Set<Character> = Set("0123456789")
    static let wordCharacters: Set<Character> = digits.union("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_")

    enum ClassMember {
        case literal(Character)
        case escape(Character)
    }

    enum Coverage: Comparable {
        case digit
        case word
        case any

        var pattern: String {
            switch self {
            case .digit: #"\d"#
            case .word: #"\w"#
            case .any: "."
            }
        }
    }

    public static func desugared(_ pattern: String) -> String? {
        let source = Array(pattern)
        var output = ""
        var index = 0
        while index < source.count {
            switch source[index] {
            case "\\":
                guard index + 1 < source.count else { return nil }
                let escaped = source[index + 1]
                if classEscapes.contains(escaped) || metacharacters.contains(escaped) {
                    output.append("\\")
                    output.append(escaped)
                } else if bareEscapes.contains(escaped) {
                    output.append(escaped)
                } else {
                    return nil
                }
                index += 2
            case "[":
                guard let (members, end) = characterClass(source, from: index),
                    let widened = widening(of: members)
                else { return nil }
                output += widened
                index = end
            case "(":
                guard let end = groupOpening(source, from: index) else { return nil }
                output += "(?:"
                index = end
            case "{":
                guard let end = quantifier(source, from: index) else { return nil }
                output += String(source[index..<end])
                index = end
            case "]":
                return nil
            case "^":
                guard index == 0 else { return nil }
                index += 1
            case "$":
                guard index == source.count - 1 else { return nil }
                index += 1
            case let character:
                output.append(character)
                index += 1
            }
        }
        return output.isEmpty ? nil : output
    }

    static func characterClass(_ source: [Character], from start: Int) -> ([ClassMember], Int)? {
        var index = start + 1
        guard index < source.count, source[index] != "^" else { return nil }
        var members: [ClassMember] = []
        while index < source.count, source[index] != "]" {
            guard let (member, next) = classMember(source, from: index) else { return nil }
            index = next
            switch member {
            case .literal(let lower)
            where index + 1 < source.count && source[index] == "-" && source[index + 1] != "]":
                guard case (.literal(let upper), let end)? = classMember(source, from: index + 1),
                    let expanded = expand(lower, upper)
                else { return nil }
                members += expanded.map(ClassMember.literal)
                index = end
            default:
                members.append(member)
            }
        }
        guard index < source.count, !members.isEmpty else { return nil }
        return (members, index + 1)
    }

    static func classMember(_ source: [Character], from start: Int) -> (ClassMember, Int)? {
        switch source[start] {
        case "[":
            return nil
        case "\\":
            guard start + 1 < source.count else { return nil }
            let escaped = source[start + 1]
            if classEscapes.contains(escaped) {
                return (.escape(escaped), start + 2)
            }
            guard metacharacters.contains(escaped) || bareEscapes.contains(escaped) else { return nil }
            return (.literal(escaped), start + 2)
        case let character:
            return (.literal(character), start + 1)
        }
    }

    static func expand(_ lower: Character, _ upper: Character) -> [Character]? {
        guard let low = lower.asciiValue, let high = upper.asciiValue, low <= high else { return nil }
        return (low...high).map { Character(UnicodeScalar($0)) }
    }

    static func groupOpening(_ source: [Character], from start: Int) -> Int? {
        guard start + 1 < source.count, source[start + 1] == "?" else { return start + 1 }
        guard start + 2 < source.count, source[start + 2] == ":" else { return nil }
        return start + 3
    }

    static func quantifier(_ source: [Character], from start: Int) -> Int? {
        guard let end = source[start...].firstIndex(of: "}") else { return nil }
        let body = source[(start + 1)..<end]
        guard !body.isEmpty, body.allSatisfy({ $0.isNumber || $0 == "," }),
            body.filter({ $0 == "," }).count <= 1, body.first != ","
        else { return nil }
        return end + 1
    }

    static func widening(of members: [ClassMember]) -> String? {
        if members.count == 1, case .literal(let character) = members[0] {
            return escaping(character)
        }
        let covered = members.compactMap(coverage(of:))
        guard covered.count == members.count else { return nil }
        return covered.max()?.pattern
    }

    static func coverage(of member: ClassMember) -> Coverage? {
        switch member {
        case .escape(let escape):
            return escape == "d" ? .digit : .word
        case .literal(let character) where character.isNewline:
            return nil
        case .literal(let character) where digits.contains(character):
            return .digit
        case .literal(let character) where wordCharacters.contains(character):
            return .word
        case .literal:
            return .any
        }
    }

    static func escaping(_ character: Character) -> String {
        metacharacters.contains(character) ? "\\\(character)" : String(character)
    }
}
