// HALMenu.swift: the HAL menu-bar item. See AUTOSTART.md.
//
// Shows at a glance whether the HAL voice session is running and whether it is
// listening, paused or muted, and offers the everyday controls. It needs no
// special permission (no Accessibility, no Automation): it reads the project's
// state files and runs the project's own commands.
//
//   HALMenu               the menu-bar item (started at login by launchd/local.hal.menubar.plist)
//   HALMenu --dry-run     the same, but a menu action prints its command line instead of running it
//   HALMenu --status      print the state read from the files, once, and exit
//   HALMenu --print-menu  print the menu for the current state, with each item's command line, and exit
//
// Build: menubar/build.sh (swiftc; AppKit only, no other dependencies).
//
// State comes from files, read every 2 s and whenever the menu opens; no process
// is started for that. Each recorded pid is checked to be alive and to be the
// right program (its arguments, read with sysctl), so a stale file or a reused
// pid is ignored.
//
//   state/listen/proxy.json        the voice session: Claude Code under the pty proxy
//   state/autostart/session.json   bin/hal-session: running, or waiting to restart after a crash
//   state/listen/listener.json     the listener: loading, listening, holding a request, paused
//   state/listen/paused            the pause flag (hal-listen pause / resume)
//   state/muted                    the mute flag (hal off / on)
//   state/speaker.pid, ack.pid     speech in progress
//
// The decisions page is offered only if 127.0.0.1:8765 accepts a connection
// when the menu opens. HAL_STATE_DIR overrides state/ as for the other tools,
// HAL_ROOT the project folder (normally the folder above this binary's), and
// HAL_DECISIONS_PORT the port.

import AppKit
import Darwin

let usage = """
    usage: HALMenu [--dry-run | --status | --print-menu]
      (no option)    run the menu-bar item
      --dry-run      run it, but menu actions print their command line instead of running it
      --status       print the state read from the files, and exit
      --print-menu   print the menu for the current state and each item's command line, and exit
    """

// MARK: - Paths

struct Paths {
    let root: URL
    let state: URL

    var listen: URL { state.appendingPathComponent("listen") }
    var autostart: URL { state.appendingPathComponent("autostart") }

    func tool(_ relative: String) -> String { root.appendingPathComponent(relative).path }

    static func current() -> Paths {
        let env = ProcessInfo.processInfo.environment
        let root: URL
        if let r = env["HAL_ROOT"], !r.isEmpty {
            root = URL(fileURLWithPath: r, isDirectory: true)
        } else {
            let exe = Bundle.main.executableURL ?? URL(fileURLWithPath: CommandLine.arguments[0])
            root = exe.resolvingSymlinksInPath().deletingLastPathComponent().deletingLastPathComponent()
        }
        let state: URL
        if let s = env["HAL_STATE_DIR"], !s.isEmpty {
            state = URL(fileURLWithPath: s, isDirectory: true)
        } else {
            state = root.appendingPathComponent("state", isDirectory: true)
        }
        return Paths(root: root, state: state)
    }
}

// MARK: - Processes

enum Proc {
    static func alive(_ pid: Int32) -> Bool {
        guard pid > 1 else { return false }
        return kill(pid, 0) == 0 || errno == EPERM
    }

    /// The process's arguments joined by spaces (sysctl KERN_PROCARGS2: argc,
    /// the executable path, padding, then the arguments, each NUL-terminated).
    static func arguments(_ pid: Int32) -> String? {
        var mib: [Int32] = [CTL_KERN, KERN_PROCARGS2, pid]
        var size = 0
        guard sysctl(&mib, 3, nil, &size, nil, 0) == 0, size > 4 else { return nil }
        var buffer = [UInt8](repeating: 0, count: size)
        guard sysctl(&mib, 3, &buffer, &size, nil, 0) == 0, size > 4 else { return nil }
        let argc = Int(buffer.withUnsafeBytes { $0.loadUnaligned(as: Int32.self) })
        var i = 4
        while i < size && buffer[i] != 0 { i += 1 }   // the executable path
        while i < size && buffer[i] == 0 { i += 1 }   // padding
        var args: [String] = []
        var start = i
        while i < size && args.count < argc {
            if buffer[i] == 0 {
                args.append(String(decoding: buffer[start..<i], as: UTF8.self))
                start = i + 1
            }
            i += 1
        }
        return args.joined(separator: " ")
    }
}

/// Is a pid alive and the expected program? Arguments are cached per pid while it lives.
final class ProcessChecker {
    private var cache: [Int32: String] = [:]

    func isRunning(_ pid: Int32, matching marks: [String]) -> Bool {
        guard Proc.alive(pid) else {
            cache[pid] = nil
            return false
        }
        let args: String
        if let cached = cache[pid] {
            args = cached
        } else {
            guard let read = Proc.arguments(pid) else { return false }
            cache[pid] = read
            args = read
        }
        return marks.allSatisfy { args.contains($0) }
    }

    func prune() {
        cache = cache.filter { Proc.alive($0.key) }
    }
}

// MARK: - State

struct Snapshot {
    enum Listening: String {
        case listening, holding, loading, paused
        case notRunning = "not running"
        case stale = "not responding"
    }

    var sessionPid: Int32?          // the pty proxy running Claude Code
    var sessionTTY: String?
    var launcherPid: Int32?         // bin/hal-session
    var launcherState: String?      // starting, running, waiting (to restart)
    var restartAt: Double?
    var restarts = 0
    var listenerPid: Int32?
    var listenerState: String?
    var listenerAge: Double?        // seconds since listener.json was updated
    var paused = false              // the pause flag
    var muted = false
    var speaking = false

    var restarting: Bool { sessionPid == nil && launcherState == "waiting" }
    var running: Bool { sessionPid != nil || launcherState == "running" || launcherState == "starting" }

    var listening: Listening {
        guard listenerPid != nil, listenerState != "stopped" else { return .notRunning }
        if paused || listenerState == "paused" { return .paused }
        // The listener rewrites its file every 2 s, but not while it loads the
        // speech model, which can take a while from a cold disk at login.
        let age = listenerAge ?? 0
        if listenerState == "loading" { return age > 120 ? .stale : .loading }
        if age > 15 { return .stale }
        return listenerState == "holding" ? .holding : .listening
    }
}

final class StateReader {
    let paths: Paths
    let checker = ProcessChecker()

    init(paths: Paths) { self.paths = paths }

    private func json(_ url: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: url),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        return object
    }

    private func pid(_ value: Any?) -> Int32? {
        guard let number = value as? NSNumber else { return nil }
        let pid = number.int32Value
        return pid > 1 ? pid : nil
    }

    private func exists(_ url: URL) -> Bool { FileManager.default.fileExists(atPath: url.path) }

    func snapshot() -> Snapshot {
        checker.prune()
        var s = Snapshot()
        if let proxy = json(paths.listen.appendingPathComponent("proxy.json")), let pid = pid(proxy["pid"]),
           checker.isRunning(pid, matching: ["hal.listener.ptyproxy"]) {
            s.sessionPid = pid
            s.sessionTTY = proxy["tty"] as? String
        }
        if let launcher = json(paths.autostart.appendingPathComponent("session.json")),
           let pid = pid(launcher["pid"]), let state = launcher["state"] as? String,
           ["starting", "running", "waiting"].contains(state),
           checker.isRunning(pid, matching: ["hal_session"]) {
            s.launcherPid = pid
            s.launcherState = state
            s.restartAt = (launcher["restart_at"] as? NSNumber)?.doubleValue
            s.restarts = (launcher["restarts"] as? NSNumber)?.intValue ?? 0
        }
        if let listener = json(paths.listen.appendingPathComponent("listener.json")), let pid = pid(listener["pid"]),
           checker.isRunning(pid, matching: ["hal.listener.cli", "_serve"]) {
            s.listenerPid = pid
            s.listenerState = listener["state"] as? String
            if let updated = (listener["updated"] as? NSNumber)?.doubleValue {
                s.listenerAge = Date().timeIntervalSince1970 - updated
            }
        }
        s.paused = exists(paths.listen.appendingPathComponent("paused"))
        s.muted = exists(paths.state.appendingPathComponent("muted"))
        for name in ["speaker.pid", "ack.pid"] {
            if let worker = json(paths.state.appendingPathComponent(name)), let pid = pid(worker["pid"]),
               checker.isRunning(pid, matching: ["_worker"]) {
                s.speaking = true
            }
        }
        return s
    }
}

// MARK: - What to show

/// The menu-bar look: SF Symbols drawn as one template image, so the menu bar
/// tints it for light and dark; short text if a symbol is missing.
struct Look: Equatable {
    var symbols: [String]
    var text: String
    var tooltip: String
}

func look(for s: Snapshot) -> Look {
    var symbols: [String]
    var text: String
    var tooltip: String
    if s.restarting {
        (symbols, text, tooltip) = (["arrow.triangle.2.circlepath"], "HAL restarting", "HAL: restarting the voice session")
    } else if !s.running {
        (symbols, text, tooltip) = (["moon.zzz"], "HAL off", "HAL: the voice session is not running")
    } else {
        switch s.listening {
        case .listening, .holding:
            (symbols, text, tooltip) = (["mic"], "HAL", "HAL: listening")
        case .loading:
            (symbols, text, tooltip) = (["hourglass"], "HAL loading", "HAL: loading the speech model")
        case .paused:
            (symbols, text, tooltip) = (["mic.slash"], "HAL paused", "HAL: listening is paused")
        case .notRunning, .stale:
            (symbols, text, tooltip) = (["mic.badge.xmark"], "HAL not listening", "HAL: running, but not listening")
        }
    }
    if s.muted {
        symbols.append("speaker.slash")
        text += ", muted"
        tooltip += "; the voice is muted"
    }
    return Look(symbols: symbols, text: text, tooltip: tooltip)
}

@MainActor
func composeImage(_ names: [String], description: String) -> NSImage? {
    let config = NSImage.SymbolConfiguration(pointSize: 14, weight: .regular)
    var parts: [NSImage] = []
    for name in names {
        guard let symbol = NSImage(systemSymbolName: name, accessibilityDescription: nil)?
            .withSymbolConfiguration(config) else { return nil }
        parts.append(symbol)
    }
    guard !parts.isEmpty else { return nil }
    let gap: CGFloat = 3
    let width = parts.reduce(0) { $0 + $1.size.width } + gap * CGFloat(parts.count - 1)
    let height = parts.map { $0.size.height }.max() ?? 16
    let image = NSImage(size: NSSize(width: ceil(width), height: ceil(height)))
    image.lockFocus()
    var x: CGFloat = 0
    for part in parts {
        part.draw(in: NSRect(x: x, y: (height - part.size.height) / 2, width: part.size.width, height: part.size.height))
        x += part.size.width + gap
    }
    image.unlockFocus()
    image.isTemplate = true
    image.accessibilityDescription = description
    return image
}

enum Action {
    case run([String])
    case quit
}

enum Entry {
    case info(String)
    case action(String, Action)
    case separator
}

func menuEntries(_ s: Snapshot, paths: Paths, decisionsAnswer: Bool, port: Int) -> [Entry] {
    var status: [Entry] = []
    if let pid = s.sessionPid {
        let tty = s.sessionTTY.map { ", " + ($0 as NSString).lastPathComponent } ?? ""
        status.append(.info("Voice session: running (pid \(pid)\(tty))"))
    } else if s.restarting {
        let wait = s.restartAt.map { " in \(max(0, Int(($0 - Date().timeIntervalSince1970).rounded()))) s" } ?? ""
        status.append(.info("Voice session: restarting\(wait) (restart \(s.restarts))"))
    } else if s.running, let pid = s.launcherPid {
        status.append(.info("Voice session: running (launcher pid \(pid))"))
    } else {
        status.append(.info("Voice session: not running"))
    }
    switch s.listening {
    case .listening: status.append(.info("Listening: yes"))
    case .holding: status.append(.info("Listening: yes; a request waits for a dialog to close"))
    case .loading: status.append(.info("Listening: loading the speech model"))
    case .paused: status.append(.info("Listening: paused"))
    case .notRunning: status.append(.info(s.paused ? "Listening: no (paused; no listener running)" : "Listening: no listener running"))
    case .stale: status.append(.info("Listening: the listener is not responding"))
    }
    status.append(.info(s.muted ? "Voice: muted" : (s.speaking ? "Voice: on, speaking now" : "Voice: on")))

    let listen = paths.tool("bin/hal-listen")
    let hal = paths.tool("bin/hal")
    let controls: [Entry] = [
        s.paused ? .action("Resume Listening", .run([listen, "resume"])) : .action("Pause Listening", .run([listen, "pause"])),
        s.muted ? .action("Unmute Voice", .run([hal, "on"])) : .action("Mute Voice", .run([hal, "off"])),
        .action("Stop Speaking Now", .run([hal, "stop"])),
    ]
    var places: [Entry] = []
    if decisionsAnswer {
        places.append(.action("Open Decisions Page", .run(["/usr/bin/open", "http://127.0.0.1:\(port)/"])))
    }
    if !s.running && !s.restarting {
        places.append(.action("Start Voice Session", .run([paths.tool("scripts/open-voice-window.sh")])))
    }
    var entries: [Entry] = []
    for group in [status, controls, places, [.action("Quit HAL Menu", .quit)]] where !group.isEmpty {
        if !entries.isEmpty { entries.append(.separator) }
        entries.append(contentsOf: group)
    }
    return entries
}

/// Does something accept a TCP connection on 127.0.0.1:port within the timeout?
func portAnswers(_ port: Int, timeoutMs: Int32 = 300) -> Bool {
    let fd = socket(AF_INET, SOCK_STREAM, 0)
    guard fd >= 0 else { return false }
    defer { close(fd) }
    _ = fcntl(fd, F_SETFL, fcntl(fd, F_GETFL, 0) | O_NONBLOCK)
    var addr = sockaddr_in()
    addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
    addr.sin_family = sa_family_t(AF_INET)
    addr.sin_port = in_port_t(UInt16(clamping: port).bigEndian)
    addr.sin_addr.s_addr = inet_addr("127.0.0.1")
    let result = withUnsafePointer(to: &addr) {
        $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
            connect(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
        }
    }
    if result == 0 { return true }
    guard errno == EINPROGRESS else { return false }
    var pfd = pollfd(fd: fd, events: Int16(POLLOUT), revents: 0)
    guard poll(&pfd, 1, timeoutMs) == 1 else { return false }
    var err: Int32 = 0
    var len = socklen_t(MemoryLayout<Int32>.size)
    guard getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &len) == 0 else { return false }
    return err == 0
}

func shellQuote(_ argv: [String]) -> String {
    let safe = Set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/-_.:=@%+,")
    return argv.map { arg in
        if !arg.isEmpty && arg.allSatisfy({ safe.contains($0) }) { return arg }
        return "'" + arg.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }.joined(separator: " ")
}

func logLine(_ message: String) {
    let stamp = ISO8601DateFormatter.string(from: Date(), timeZone: .current, formatOptions: [.withInternetDateTime])
    print("\(stamp) HALMenu: \(message)")
}

// MARK: - The menu-bar item

final class ActionBox: NSObject {
    let action: Action
    init(_ action: Action) { self.action = action }
}

@MainActor
final class MenuController: NSObject, NSMenuDelegate {
    let paths: Paths
    let reader: StateReader
    let dryRun: Bool
    let port: Int
    let item: NSStatusItem
    let menu = NSMenu()
    private var shown: Look?
    private var timer: Timer?

    init(paths: Paths, dryRun: Bool, port: Int) {
        self.paths = paths
        self.reader = StateReader(paths: paths)
        self.dryRun = dryRun
        self.port = port
        self.item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        super.init()
        menu.autoenablesItems = false
        menu.delegate = self
        item.menu = menu
        refresh()
        let timer = Timer(timeInterval: 2, target: self, selector: #selector(tick), userInfo: nil, repeats: true)
        timer.tolerance = 0.5
        RunLoop.main.add(timer, forMode: .common)
        self.timer = timer
    }

    @objc func tick() { refresh() }

    @discardableResult
    func refresh() -> Snapshot {
        let s = reader.snapshot()
        let l = look(for: s)
        if l != shown {
            shown = l
            if let button = item.button {
                if let image = composeImage(l.symbols, description: l.tooltip) {
                    button.image = image
                    button.title = ""
                    button.imagePosition = .imageOnly
                } else {
                    button.image = nil
                    button.title = l.text
                }
                button.toolTip = l.tooltip
                button.setAccessibilityLabel(l.tooltip)
            }
        }
        return s
    }

    func menuNeedsUpdate(_ menu: NSMenu) {
        let s = refresh()
        menu.removeAllItems()
        for entry in menuEntries(s, paths: paths, decisionsAnswer: portAnswers(port), port: port) {
            switch entry {
            case .separator:
                menu.addItem(.separator())
            case .info(let text):
                let mi = NSMenuItem(title: text, action: nil, keyEquivalent: "")
                mi.isEnabled = false
                menu.addItem(mi)
            case .action(let title, let action):
                let mi = NSMenuItem(title: title, action: #selector(choose(_:)), keyEquivalent: "")
                mi.target = self
                mi.representedObject = ActionBox(action)
                menu.addItem(mi)
            }
        }
    }

    @objc func choose(_ sender: NSMenuItem) {
        guard let box = sender.representedObject as? ActionBox else { return }
        switch box.action {
        case .quit:
            if dryRun { logLine("dry-run: quit") }
            NSApp.terminate(nil)
        case .run(let argv):
            run(argv)
        }
    }

    private func run(_ argv: [String]) {
        if dryRun {
            logLine("dry-run: would run: \(shellQuote(argv))")
            return
        }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: argv[0])
        process.arguments = Array(argv.dropFirst())
        process.currentDirectoryURL = paths.root
        process.standardInput = FileHandle.nullDevice
        process.standardOutput = FileHandle.nullDevice
        let line = shellQuote(argv)
        process.terminationHandler = { [weak self] finished in
            let code = finished.terminationStatus
            DispatchQueue.main.async {
                MainActor.assumeIsolated {
                    if code != 0 { logLine("command failed (exit \(code)): \(line)") }
                    _ = self?.refresh()
                }
            }
        }
        do {
            try process.run()
        } catch {
            logLine("cannot run \(line): \(error.localizedDescription)")
        }
        // The listener notices a pause or resume within half a second: look again then.
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) { [weak self] in
            MainActor.assumeIsolated { _ = self?.refresh() }
        }
    }
}

// MARK: - Command-line modes

@MainActor
func printStatus(_ paths: Paths) {
    let s = StateReader(paths: paths).snapshot()
    let l = look(for: s)
    print("project:   \(paths.root.path)")
    print("state:     \(paths.state.path)")
    if let pid = s.sessionPid {
        print("session:   running (pty proxy pid \(pid)\(s.sessionTTY.map { ", " + $0 } ?? ""))")
    } else if s.restarting {
        print("session:   restarting (restart \(s.restarts))")
    } else if s.running {
        print("session:   running (under bin/hal-session, no pty proxy)")
    } else {
        print("session:   not running")
    }
    print("launcher:  " + (s.launcherPid.map { "bin/hal-session pid \($0), \(s.launcherState ?? "?")" } ?? "none"))
    if let pid = s.listenerPid {
        let age = s.listenerAge.map { String(format: ", updated %.0f s ago", $0) } ?? ""
        print("listener:  pid \(pid), state \(s.listenerState ?? "?")\(age)")
    } else {
        print("listener:  not running")
    }
    print("listening: \(s.listening.rawValue)")
    print("paused:    \(s.paused ? "yes" : "no") (flag state/listen/paused)")
    print("muted:     \(s.muted ? "yes" : "no") (flag state/muted)")
    print("speaking:  \(s.speaking ? "yes" : "no")")
    let found = composeImage(l.symbols, description: l.tooltip) != nil
    print("icon:      \(l.symbols.joined(separator: " + ")) (\(found ? "symbols found" : "symbol missing: shows text \"\(l.text)\""))")
    print("tooltip:   \(l.tooltip)")
}

func printMenu(_ paths: Paths, port: Int) {
    let s = StateReader(paths: paths).snapshot()
    let answers = portAnswers(port)
    print("decisions page 127.0.0.1:\(port): \(answers ? "answers" : "does not answer")")
    for entry in menuEntries(s, paths: paths, decisionsAnswer: answers, port: port) {
        switch entry {
        case .separator: print("  ---")
        case .info(let text): print("  [info] \(text)")
        case .action(let title, .run(let argv)): print("  \(title)  ->  \(shellQuote(argv))")
        case .action(let title, .quit): print("  \(title)  ->  (quits the menu-bar item)")
        }
    }
    // The items the other states would show instead, so every command can be checked.
    var other = s
    other.paused.toggle()
    other.muted.toggle()
    other.sessionPid = nil
    other.launcherState = nil
    print("in other states:")
    for entry in menuEntries(other, paths: paths, decisionsAnswer: true, port: port) {
        if case .action(let title, .run(let argv)) = entry, title != "Stop Speaking Now",
           !menuEntries(s, paths: paths, decisionsAnswer: answers, port: port).contains(where: {
               if case .action(let t, _) = $0 { return t == title } else { return false }
           }) {
            print("  \(title)  ->  \(shellQuote(argv))")
        }
    }
}

// MARK: - Main

setvbuf(stdout, nil, _IOLBF, 0)  // line by line, even into launchd's log file
let arguments = Array(CommandLine.arguments.dropFirst())
let paths = Paths.current()
let port = Int(ProcessInfo.processInfo.environment["HAL_DECISIONS_PORT"] ?? "") ?? 8765

if arguments.contains("-h") || arguments.contains("--help") {
    print(usage)
    exit(0)
}
if let unknown = arguments.first(where: { !["--dry-run", "--status", "--print-menu"].contains($0) }) {
    FileHandle.standardError.write("HALMenu: unknown option \(unknown)\n\(usage)\n".data(using: .utf8)!)
    exit(2)
}
if arguments.contains("--status") {
    MainActor.assumeIsolated { printStatus(paths) }
    exit(0)
}
if arguments.contains("--print-menu") {
    printMenu(paths, port: port)
    exit(0)
}
guard FileManager.default.isExecutableFile(atPath: paths.tool("bin/hal")) else {
    // Exit 0, so launchd doesn't start it again and again: this won't fix itself.
    logLine("no HAL project at \(paths.root.path) (no bin/hal); not starting")
    exit(0)
}

// One menu-bar item at a time (exit 0, so launchd doesn't start it again).
try? FileManager.default.createDirectory(at: paths.autostart, withIntermediateDirectories: true)
let lockFD = open(paths.autostart.appendingPathComponent("menubar.lock").path, O_RDWR | O_CREAT | O_CLOEXEC, 0o600)
if lockFD < 0 || flock(lockFD, LOCK_EX | LOCK_NB) != 0 {
    logLine("another HAL menu-bar item is already running; not starting")
    exit(0)
}

let dryRun = arguments.contains("--dry-run")
logLine("started\(dryRun ? " (dry run: menu actions are printed, not run)" : "") for \(paths.root.path)")
MainActor.assumeIsolated {
    let app = NSApplication.shared
    app.setActivationPolicy(.accessory)
    let controller = MenuController(paths: paths, dryRun: dryRun, port: port)
    withExtendedLifetime(controller) { app.run() }
}
