import Testing

@testable import SpawnllmApple

@Suite("pattern desugaring")
struct RegexPatternTests {
    @Test(
        "widens a class to the narrowest escape covering it",
        arguments: [
            (#"^[A-Z]{3}-\d{4}$"#, #"\w{3}-\d{4}"#),
            ("[0-9]", #"\d"#),
            ("[0-9]{4}", #"\d{4}"#),
            ("[13579]", #"\d"#),
            (#"[\d]"#, #"\d"#),
            ("[abc]", #"\w"#),
            ("[a-c0-9_]", #"\w"#),
            (#"[\d\w]"#, #"\w"#),
            (#"[a-z\d]"#, #"\w"#),
            ("[a-z!]", "."),
            ("[.-]", "."),
            ("[a]", "a"),
            ("[5]", "5"),
            ("[.]", #"\."#),
        ]
    )
    func widens(pattern: String, expected: String) {
        #expect(RegexPattern.desugared(pattern) == expected)
    }

    @Test(
        "rewrites into the accepted subset",
        arguments: [
            ("^hello$", "hello"),
            (#"\d{3}\.\d{2}"#, #"\d{3}\.\d{2}"#),
            ("(cat|dog)", "(?:cat|dog)"),
            ("(?:cat|dog)", "(?:cat|dog)"),
            ("a.*b+c?", "a.*b+c?"),
            (#"a\-b"#, "a-b"),
            (#"\w+@\w+"#, #"\w+@\w+"#),
            ("x{2,}", "x{2,}"),
            ("x{2,4}", "x{2,4}"),
        ]
    )
    func desugars(pattern: String, expected: String) {
        #expect(RegexPattern.desugared(pattern) == expected)
    }

    @Test(
        "drops a pattern outside the subset",
        arguments: [
            "[^abc]",
            "[[:alpha:]]",
            #"\p{L}+"#,
            "(?=lookahead)",
            "(?<name>x)",
            #"(a)\1"#,
            "a^b",
            "a$b",
            "]",
            "[unterminated",
            #"trailing\"#,
            #"\s+"#,
            "[]",
            "x{a}",
            "x{,3}",
            "^$",
            "[z-a]",
            "[a-é]",
            "[a\n]",
        ]
    )
    func drops(pattern: String) {
        #expect(RegexPattern.desugared(pattern) == nil)
    }

    @Test("every desugared pattern compiles as a regex")
    func compiles() throws {
        for pattern in [#"^[A-Z]{3}-\d{4}$"#, "[a-c0-9_]", "(cat|dog)", "[.-]", #"\w+@\w+"#, "[0-9]{4}", "[.]"] {
            let desugared = try #require(RegexPattern.desugared(pattern))
            #expect(throws: Never.self) { try Regex(desugared) }
        }
    }

    @Test("a widened class keeps the shape it constrained and admits everything it matched")
    func widensWithoutLosingShape() throws {
        let regex = try Regex(#require(RegexPattern.desugared(#"^[A-Z]{3}-\d{4}$"#)))
        #expect("XKR-8821".wholeMatch(of: regex) != nil)
        #expect("xkr-8821".wholeMatch(of: regex) != nil)
        #expect("XKR-882".wholeMatch(of: regex) == nil)
        #expect("XKR8821".wholeMatch(of: regex) == nil)
        #expect("XKR-882A".wholeMatch(of: regex) == nil)
    }
}
