import Foundation
import SpawnllmApple

@main
struct Entry {
    static func main() async {
        guard !CommandLine.arguments.contains("--probe") else {
            let probe = Sidecar.probe()
            FileHandle.standardOutput.write(probe.line())
            exit(probe.available ? 0 : 1)
        }
        let reply: Reply
        do {
            reply = await Sidecar.generate(try Request.decode(FileHandle.standardInput.readDataToEndOfFile()))
        } catch {
            reply = SidecarError.resolve(error).reply
        }
        FileHandle.standardOutput.write(reply.line())
    }
}
