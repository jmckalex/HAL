// hal-duplex: the listener's microphone with Apple's echo cancellation, and the
// voice's playback through the same audio engine. See LISTENER.md, "Interrupting
// the voice".
//
//   hal-duplex serve [--socket PATH] [--mode auto|vp|raw] [--ducking min|default|mid|max] [--agc]
//                    [--yield-to-other-input] [--debug-levels]
//   hal-duplex play FILE [--socket PATH]
//   hal-duplex probe
//
// serve (run by the listener as its child; it exits when its stdin closes)
//   Captures the system's current default input with voice processing (Apple's
//   VoiceProcessingIO, through AVAudioEngine) and writes 16 kHz mono float32
//   frames of 512 samples to stdout. On macOS the echo canceller removes
//   whatever the output device plays, from any process (Apple's header: "any of
//   the audio that is played from the device"; measured on an M4 Max, macOS 15.6,
//   for afplay as for this engine). The voice is still played here, by workers
//   running `hal-duplex play FILE`: here it is not ducked as "other audio", it
//   can be paused and resumed at once, and its position is known. Killing the
//   client stops it.
//   The default input and output are followed: when either changes (AirPods
//   connect, headphones are plugged in) the engine is rebuilt on the new
//   devices. With headphones as the output there is no echo to cancel, so in
//   mode "auto" voice processing is left off there (no ducking, no processing
//   of the microphone). With --yield-to-other-input it is also left off while
//   another process captures audio (a call): voice processing makes other
//   clients of the microphone much quieter.
//
//   stdout records: one type byte, a little-endian UInt32 length, the payload.
//     'A'  512 float32 samples, then float32 level of what this engine rendered
//          (RMS over the last render buffer), then float32 1 if the voice is
//          playing here, else 0
//     'E'  a JSON event: ready, error, device, play, paused, resumed, played, status
//   stdin commands, one per line: pause | resume | stop | status | quit
//
// play
//   Connects to the socket, asks serve to play FILE and waits until it has been
//   played. If nothing answers (no listener, or one without this helper), it
//   runs /usr/bin/afplay FILE instead, so the voice is never lost.
//
// probe
//   Prints the default devices as JSON, with whether the output counts as
//   headphones and which other processes capture audio. Nothing is opened.
//
// Nothing is written to disk. Build: duplex/build.sh (swiftc, no other dependencies).

import AVFoundation
import CoreAudio
import Darwin
import Foundation

let rate = 16000.0
let frameSamples = 512

// MARK: - output to the listener

let outLock = NSLock()

func writeAll(_ fd: Int32, _ data: Data) -> Bool {
    return data.withUnsafeBytes { (p: UnsafeRawBufferPointer) -> Bool in
        guard let base = p.baseAddress else { return true }
        var off = 0
        while off < p.count {
            let n = Darwin.write(fd, base + off, p.count - off)
            if n < 0 {
                if errno == EINTR { continue }
                return false
            }
            off += n
        }
        return true
    }
}

func record(_ type: UInt8, _ payload: Data) {
    var d = Data(capacity: payload.count + 5)
    d.append(type)
    var n = UInt32(payload.count).littleEndian
    withUnsafeBytes(of: &n) { d.append(contentsOf: $0) }
    d.append(payload)
    outLock.lock()
    let ok = writeAll(1, d)
    outLock.unlock()
    if !ok { exit(0) }   // the listener has gone
}

func event(_ name: String, _ fields: [String: Any] = [:]) {
    var f = fields
    f["event"] = name
    if let d = try? JSONSerialization.data(withJSONObject: f, options: [.sortedKeys]) {
        record(0x45, d)
    }
}

func note(_ s: String) {
    FileHandle.standardError.write(("hal-duplex: " + s + "\n").data(using: .utf8)!)
}

// MARK: - devices

func systemProperty(_ selector: AudioObjectPropertySelector) -> AudioDeviceID {
    var id = AudioDeviceID(0)
    var size = UInt32(MemoryLayout<AudioDeviceID>.size)
    var addr = AudioObjectPropertyAddress(mSelector: selector, mScope: kAudioObjectPropertyScopeGlobal,
                                          mElement: kAudioObjectPropertyElementMain)
    let st = AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size, &id)
    return st == noErr ? id : 0
}

func deviceName(_ id: AudioDeviceID) -> String {
    guard id != 0 else { return "" }
    var addr = AudioObjectPropertyAddress(mSelector: kAudioObjectPropertyName, mScope: kAudioObjectPropertyScopeGlobal,
                                          mElement: kAudioObjectPropertyElementMain)
    var name: Unmanaged<CFString>?
    var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    let st = AudioObjectGetPropertyData(id, &addr, 0, nil, &size, &name)
    guard st == noErr, let n = name else { return "?" }
    return n.takeRetainedValue() as String
}

func uint32Property(_ id: AudioDeviceID, _ selector: AudioObjectPropertySelector,
                    _ scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal) -> UInt32? {
    var v = UInt32(0)
    var size = UInt32(MemoryLayout<UInt32>.size)
    var addr = AudioObjectPropertyAddress(mSelector: selector, mScope: scope, mElement: kAudioObjectPropertyElementMain)
    guard AudioObjectHasProperty(id, &addr) else { return nil }
    return AudioObjectGetPropertyData(id, &addr, 0, nil, &size, &v) == noErr ? v : nil
}

func hasStreams(_ id: AudioDeviceID, _ scope: AudioObjectPropertyScope) -> Bool {
    var addr = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyStreams, mScope: scope,
                                          mElement: kAudioObjectPropertyElementMain)
    var size = UInt32(0)
    return AudioObjectGetPropertyDataSize(id, &addr, 0, nil, &size) == noErr && size > 0
}

func fourCC(_ v: UInt32) -> String {
    let bytes = [24, 16, 8, 0].map { UInt8((v >> UInt32($0)) & 0xff) }
    return String(bytes: bytes, encoding: .ascii) ?? "\(v)"
}

/// Is the output device headphones (so the microphone can't hear the voice)?
/// Built-in output with its headphone data source; Bluetooth that is a headset
/// (it also has a microphone) or is named like headphones; USB named so.
func outputKind(_ id: AudioDeviceID) -> (headphones: Bool, kind: String) {
    guard id != 0 else { return (false, "none") }
    let transport = uint32Property(id, kAudioDevicePropertyTransportType) ?? 0
    let name = deviceName(id).lowercased()
    let pattern = ["airpods", "beats", "headphone", "headset", "earbud", "earphone", "buds", "powerbeats"]
    let named = pattern.contains { name.contains($0) }
    switch transport {
    case kAudioDeviceTransportTypeBuiltIn:
        let ds = uint32Property(id, kAudioDevicePropertyDataSource, kAudioDevicePropertyScopeOutput) ?? 0
        return ds == 0x6864_706E ? (true, "built-in headphones") : (false, "built-in speakers")   // 'hdpn'
    case kAudioDeviceTransportTypeBluetooth, kAudioDeviceTransportTypeBluetoothLE:
        if named { return (true, "Bluetooth headphones") }
        if hasStreams(id, kAudioDevicePropertyScopeInput) { return (true, "Bluetooth headset") }
        return (false, "Bluetooth speaker")
    case kAudioDeviceTransportTypeUSB:
        return named ? (true, "USB headphones") : (false, "USB output")
    default:
        return named ? (true, "headphones (\(fourCC(transport)))") : (false, "output (\(fourCC(transport)))")
    }
}

func deviceInfo() -> [String: Any] {
    let i = systemProperty(kAudioHardwarePropertyDefaultInputDevice)
    let o = systemProperty(kAudioHardwarePropertyDefaultOutputDevice)
    let k = outputKind(o)
    return ["input": deviceName(i), "output": deviceName(o), "output_kind": k.kind, "headphones": k.headphones,
            "input_transport": fourCC(uint32Property(i, kAudioDevicePropertyTransportType) ?? 0),
            "output_transport": fourCC(uint32Property(o, kAudioDevicePropertyTransportType) ?? 0)]
}

/// System processes that capture input because of voice processing itself, not
/// another app: corespeechd starts while voice processing runs (seen on macOS 15.6),
/// so yielding to it would turn voice processing off and on in a loop.
let notOtherApps: Set<String> = ["com.apple.CoreSpeech"]

/// Other processes capturing audio input right now (macOS 14+): [(pid, bundle id)].
func otherInputProcesses() -> [(Int32, String)] {
    var addr = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyProcessObjectList,
                                          mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
    var size = UInt32(0)
    let system = AudioObjectID(kAudioObjectSystemObject)
    guard AudioObjectGetPropertyDataSize(system, &addr, 0, nil, &size) == noErr, size > 0 else { return [] }
    var ids = [AudioObjectID](repeating: 0, count: Int(size) / MemoryLayout<AudioObjectID>.size)
    guard AudioObjectGetPropertyData(system, &addr, 0, nil, &size, &ids) == noErr else { return [] }
    let me = getpid()
    var out: [(Int32, String)] = []
    for id in ids {
        guard uint32Property(id, kAudioProcessPropertyIsRunningInput) == 1 else { continue }
        var pid = Int32(0)
        var psize = UInt32(MemoryLayout<Int32>.size)
        var paddr = AudioObjectPropertyAddress(mSelector: kAudioProcessPropertyPID, mScope: kAudioObjectPropertyScopeGlobal,
                                               mElement: kAudioObjectPropertyElementMain)
        guard AudioObjectGetPropertyData(id, &paddr, 0, nil, &psize, &pid) == noErr, pid != me else { continue }
        var baddr = AudioObjectPropertyAddress(mSelector: kAudioProcessPropertyBundleID, mScope: kAudioObjectPropertyScopeGlobal,
                                               mElement: kAudioObjectPropertyElementMain)
        var bundle: Unmanaged<CFString>?
        var bsize = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
        let name = AudioObjectGetPropertyData(id, &baddr, 0, nil, &bsize, &bundle) == noErr
            ? (bundle?.takeRetainedValue() as String? ?? "") : ""
        if notOtherApps.contains(name) { continue }
        out.append((pid, name))
    }
    return out
}

// MARK: - playback

final class Playback {
    let id: Int
    let path: String
    let buffer: AVAudioPCMBuffer      // the whole file, in the player's format
    let seconds: Double
    var offset: AVAudioFramePosition = 0   // where the scheduled segment starts
    var paused = false
    var finished = false
    var startedAt = Date()            // when the current segment was scheduled
    var client: Int32                 // socket of the `play` client, -1 when gone
    var pausedAt = Date.distantPast

    init(id: Int, path: String, buffer: AVAudioPCMBuffer, client: Int32) {
        self.id = id
        self.path = path
        self.buffer = buffer
        self.seconds = Double(buffer.frameLength) / buffer.format.sampleRate
        self.client = client
    }
}

func loadBuffer(_ path: String, as format: AVAudioFormat) throws -> AVAudioPCMBuffer {
    let file = try AVAudioFile(forReading: URL(fileURLWithPath: path))
    let src = file.processingFormat
    guard let raw = AVAudioPCMBuffer(pcmFormat: src, frameCapacity: AVAudioFrameCount(max(file.length, 1))) else {
        throw NSError(domain: "hal-duplex", code: 1, userInfo: [NSLocalizedDescriptionKey: "cannot allocate"])
    }
    try file.read(into: raw)
    guard let conv = AVAudioConverter(from: src, to: format) else {
        throw NSError(domain: "hal-duplex", code: 2, userInfo: [NSLocalizedDescriptionKey: "cannot convert \(src)"])
    }
    let cap = AVAudioFrameCount(Double(raw.frameLength) * format.sampleRate / src.sampleRate) + 4096
    guard let out = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: cap) else {
        throw NSError(domain: "hal-duplex", code: 3, userInfo: [NSLocalizedDescriptionKey: "cannot allocate"])
    }
    var given = false
    var err: NSError?
    let status = conv.convert(to: out, error: &err) { _, st in
        if given {
            st.pointee = .endOfStream
            return nil
        }
        given = true
        st.pointee = .haveData
        return raw
    }
    if status == .error { throw err ?? NSError(domain: "hal-duplex", code: 4) }
    return out
}

// MARK: - the engine

final class Duplex {
    let ctl = DispatchQueue(label: "hal-duplex.control")
    let socketPath: String?
    let mode: String
    let ducking: String
    let agc: Bool

    // Owned by ctl.
    var engine: AVAudioEngine?
    var player: AVAudioPlayerNode?
    var vpActive = false
    var headphones = false
    var current: Playback?
    var token = 0              // invalidates completion handlers of an older schedule
    var nextId = 1
    var rebuildAt: Date?
    var failures = 0
    var builtInput = AudioDeviceID(0)
    var builtOutput = AudioDeviceID(0)
    var builds: [Date] = []
    var restarts: [Date] = []
    var yieldToOthers = false      // drop voice processing while another app captures audio
    var yielding = false

    // Capture side (tap thread), guarded by capLock.
    let capLock = NSLock()
    var converter: AVAudioConverter?
    var monoFormat: AVAudioFormat?
    var pending: [Float] = []
    let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: rate, channels: 1, interleaved: false)!

    // Render level (render tap -> capture), guarded by levelLock.
    let levelLock = NSLock()
    var renderRMS: Float = 0
    var renderAt = Date.distantPast
    var playingFlag = false

    let playFormat = AVAudioFormat(standardFormatWithSampleRate: 48000, channels: 1)!

    init(socketPath: String?, mode: String, ducking: String, agc: Bool) {
        self.socketPath = socketPath
        self.mode = mode
        self.ducking = ducking
        self.agc = agc
    }

    // ---- capture

    var debugSums: [Float] = []
    var debugCount = 0
    var debugLevels = false

    func captured(_ buf: AVAudioPCMBuffer) {
        let n = Int(buf.frameLength)
        guard n > 0, let src = buf.floatChannelData else { return }
        if debugLevels && !buf.format.isInterleaved {
            let chans = Int(buf.format.channelCount)
            if debugSums.count != chans { debugSums = [Float](repeating: 0, count: chans); debugCount = 0 }
            for c in 0..<chans { for i in 0..<n { let v = src[c][i]; debugSums[c] += v * v } }
            debugCount += n
            if debugCount >= Int(buf.format.sampleRate) / 4 {
                let dbs = debugSums.map { 10 * log10(Double($0) / Double(debugCount) + 1e-18) }
                event("levels", ["db": dbs.map { (($0 * 10).rounded()) / 10 }])
                debugSums = [Float](repeating: 0, count: chans)
                debugCount = 0
            }
        }
        capLock.lock()
        defer { capLock.unlock() }
        guard let conv = converter, let mono = monoFormat,
              let m = AVAudioPCMBuffer(pcmFormat: mono, frameCapacity: AVAudioFrameCount(n)) else { return }
        m.frameLength = AVAudioFrameCount(n)
        let dst = m.floatChannelData![0]
        if buf.format.isInterleaved {
            let stride = Int(buf.format.channelCount)
            for i in 0..<n { dst[i] = src[0][i * stride] }
        } else {
            dst.update(from: src[0], count: n)
        }
        let cap = AVAudioFrameCount(Double(n) * rate / mono.sampleRate) + 64
        guard let out = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: cap) else { return }
        var given = false
        var err: NSError?
        let status = conv.convert(to: out, error: &err) { _, st in
            if given {
                st.pointee = .noDataNow
                return nil
            }
            given = true
            st.pointee = .haveData
            return m
        }
        if status == .error || out.frameLength == 0 { return }
        let p = out.floatChannelData![0]
        pending.append(contentsOf: UnsafeBufferPointer(start: p, count: Int(out.frameLength)))
        while pending.count >= frameSamples {
            levelLock.lock()
            let level: Float = Date().timeIntervalSince(renderAt) < 0.12 ? renderRMS : 0
            let playing: Float = playingFlag ? 1 : 0
            levelLock.unlock()
            var payload = Data(capacity: (frameSamples + 2) * 4)
            pending.withUnsafeBufferPointer { payload.append(UnsafeBufferPointer(start: $0.baseAddress, count: frameSamples)) }
            let extra: [Float] = [level, playing]
            extra.withUnsafeBufferPointer { payload.append($0) }
            pending.removeFirst(frameSamples)
            record(0x41, payload)
        }
    }

    func rendered(_ buf: AVAudioPCMBuffer) {
        let n = Int(buf.frameLength)
        guard n > 0, let src = buf.floatChannelData else { return }
        var sum: Float = 0
        let ch = src[0]
        let stride = buf.format.isInterleaved ? Int(buf.format.channelCount) : 1
        for i in 0..<n { let v = ch[i * stride]; sum += v * v }
        levelLock.lock()
        renderRMS = (sum / Float(n)).squareRoot()
        renderAt = Date()
        levelLock.unlock()
    }

    // ---- engine lifecycle (on ctl)

    func duckingLevel() -> Int {
        switch ducking {
        case "default": return 0
        case "mid": return 20
        case "max": return 30
        default: return 10
        }
    }

    func build(_ reason: String) {
        teardown()
        let inDev = systemProperty(kAudioHardwarePropertyDefaultInputDevice)
        let outDev = systemProperty(kAudioHardwarePropertyDefaultOutputDevice)
        if inDev == 0 {
            fail("no default input device")
            return
        }
        let kind = outputKind(outDev)
        let others = yieldToOthers ? otherInputProcesses() : []
        yielding = !others.isEmpty
        let wantVP = (mode == "vp" || (mode == "auto" && !kind.headphones)) && !yielding
        let e = AVAudioEngine()
        var vp = false
        var vpError = ""
        if wantVP {
            do {
                try e.inputNode.setVoiceProcessingEnabled(true)
                vp = e.inputNode.isVoiceProcessingEnabled
                if !vp { vpError = "voice processing did not turn on" }
            } catch {
                vpError = error.localizedDescription
            }
            if vp {
                e.inputNode.isVoiceProcessingAGCEnabled = agc
                if #available(macOS 14.0, *) {
                    let level = AVAudioVoiceProcessingOtherAudioDuckingConfiguration.Level(rawValue: duckingLevel()) ?? .min
                    e.inputNode.voiceProcessingOtherAudioDuckingConfiguration =
                        AVAudioVoiceProcessingOtherAudioDuckingConfiguration(enableAdvancedDucking: false, duckingLevel: level)
                }
            }
        }
        let fmt = e.inputNode.outputFormat(forBus: 0)
        if fmt.sampleRate <= 0 || fmt.channelCount == 0 {
            fail("the input has no audio format (no microphone permission, or no input device)")
            return
        }
        guard let mono = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: fmt.sampleRate, channels: 1,
                                       interleaved: false),
              let conv = AVAudioConverter(from: mono, to: target) else {
            fail("cannot convert from \(fmt)")
            return
        }
        capLock.lock()
        monoFormat = mono
        converter = conv
        pending.removeAll()
        capLock.unlock()
        // The mixer must feed the output at the output device's own format: left to
        // itself the engine picks 44.1 kHz, and voice processing then fails to
        // initialise (-10875).
        let hw = e.outputNode.outputFormat(forBus: 0)
        let mixOut = hw.sampleRate > 0 && hw.channelCount > 0 ? hw
            : AVAudioFormat(standardFormatWithSampleRate: 48000, channels: 2)!
        e.connect(e.mainMixerNode, to: e.outputNode, format: mixOut)
        let p = AVAudioPlayerNode()
        e.attach(p)
        e.connect(p, to: e.mainMixerNode, format: playFormat)
        e.inputNode.installTap(onBus: 0, bufferSize: 1024, format: fmt) { [weak self] b, _ in self?.captured(b) }
        e.mainMixerNode.installTap(onBus: 0, bufferSize: 1024, format: nil) { [weak self] b, _ in self?.rendered(b) }
        e.prepare()
        do {
            try e.start()
        } catch {
            e.inputNode.removeTap(onBus: 0)
            e.mainMixerNode.removeTap(onBus: 0)
            fail("the audio engine would not start: \(error.localizedDescription)")
            return
        }
        engine = e
        player = p
        builtInput = inDev
        builtOutput = outDev
        vpActive = vp
        headphones = kind.headphones
        failures = 0
        var info = deviceInfo()
        info["reason"] = reason
        info["vp"] = vp
        info["vp_wanted"] = wantVP
        info["vp_error"] = yielding ? "another app is using audio input" : vpError
        info["other_input"] = others.map { $0.1.isEmpty ? "pid \($0.0)" : $0.1 }
        info["agc"] = vp && agc
        info["ducking"] = vp ? ducking : "none"
        info["rate"] = fmt.sampleRate
        info["channels"] = Int(fmt.channelCount)
        event("ready", info)
        if let c = current, !c.finished, !c.paused { schedule(c) }   // carry on after a device change
    }

    func fail(_ why: String) {
        failures += 1
        event("error", ["why": why, "failures": failures])
        note(why)
        // Try again on the next device change, or shortly.
        requestRebuild("retry after an error", after: min(10.0, 1.0 * Double(failures)))
    }

    func teardown() {
        guard let e = engine else { return }
        if let c = current, !c.finished, !c.paused {
            c.offset = position(c)
        }
        token += 1
        e.inputNode.removeTap(onBus: 0)
        e.mainMixerNode.removeTap(onBus: 0)
        player?.stop()
        e.stop()
        engine = nil
        player = nil
        setPlaying(false)
    }

    func requestRebuild(_ reason: String, after: Double = 0.4) {
        // A device that keeps reconfiguring must not make the engine restart in a loop.
        let now = Date()
        builds = builds.filter { now.timeIntervalSince($0) < 30 }
        let delay = builds.count >= 6 ? max(after, 5.0) : after
        builds.append(now)
        let when = now.addingTimeInterval(delay)
        let after = delay
        rebuildAt = when
        ctl.asyncAfter(deadline: .now() + after) { [weak self] in
            guard let self = self, let at = self.rebuildAt, at <= Date() else { return }
            self.rebuildAt = nil
            self.build(reason)
        }
    }

    // ---- playback (on ctl)

    func setPlaying(_ on: Bool) {
        levelLock.lock()
        playingFlag = on
        levelLock.unlock()
    }

    func position(_ c: Playback) -> AVAudioFramePosition {
        let total = AVAudioFramePosition(c.buffer.frameLength)
        if c.paused { return min(c.offset, total) }
        guard let p = player, p.isPlaying, let nt = p.lastRenderTime, nt.isSampleTimeValid,
              let pt = p.playerTime(forNodeTime: nt) else {
            // The engine has stopped: estimate from the clock.
            let elapsed = AVAudioFramePosition(Date().timeIntervalSince(c.startedAt) * c.buffer.format.sampleRate)
            return min(c.offset + max(0, elapsed), total)
        }
        return min(c.offset + max(0, pt.sampleTime), total)
    }

    func seconds(_ c: Playback, _ frames: AVAudioFramePosition) -> Double {
        return Double(frames) / c.buffer.format.sampleRate
    }

    func schedule(_ c: Playback) {
        guard let p = player else { return }
        let total = AVAudioFramePosition(c.buffer.frameLength)
        let start = min(c.offset, total)
        let count = AVAudioFrameCount(total - start)
        if count == 0 {
            finish(c, "end", start)
            return
        }
        guard let seg = AVAudioPCMBuffer(pcmFormat: c.buffer.format, frameCapacity: count) else { return }
        seg.frameLength = count
        seg.floatChannelData![0].update(from: c.buffer.floatChannelData![0] + Int(start), count: Int(count))
        token += 1
        let mine = token
        c.startedAt = Date()
        p.scheduleBuffer(seg, at: nil, options: [], completionCallbackType: .dataPlayedBack) { [weak self] _ in
            guard let self = self else { return }
            self.ctl.async {
                if self.token == mine, let cur = self.current, cur === c, !c.finished, !c.paused {
                    self.finish(c, "end", AVAudioFramePosition(c.buffer.frameLength))
                }
            }
        }
        p.play()
        setPlaying(true)
    }

    func finish(_ c: Playback, _ reason: String, _ at: AVAudioFramePosition) {
        guard !c.finished else { return }
        c.finished = true
        if current === c {
            current = nil
            setPlaying(false)
        }
        event("played", ["id": c.id, "reason": reason, "position": seconds(c, at), "seconds": c.seconds,
                         "path": c.path])
        if c.client >= 0 {
            _ = writeAll(c.client, "done \(reason) \(String(format: "%.3f", seconds(c, at)))\n".data(using: .utf8)!)
            shutdown(c.client, SHUT_WR)
        }
    }

    func stopCurrent(_ reason: String) {
        guard let c = current else { return }
        let at = c.paused ? c.offset : position(c)
        token += 1
        player?.stop()
        finish(c, reason, at)
    }

    /// A `play` client asked for path. Returns the reply line.
    func play(_ path: String, client: Int32) -> String {
        guard engine != nil, player != nil else { return "error the audio engine is not running" }
        let buffer: AVAudioPCMBuffer
        do {
            buffer = try loadBuffer(path, as: playFormat)
        } catch {
            return "error cannot read the file: \(error.localizedDescription)"
        }
        stopCurrent("replaced")
        let c = Playback(id: nextId, path: path, buffer: buffer, client: client)
        nextId += 1
        current = c
        schedule(c)
        event("play", ["id": c.id, "seconds": c.seconds, "path": path])
        return "started \(c.id) \(String(format: "%.3f", c.seconds))"
    }

    func clientGone(_ fd: Int32) {
        if let c = current, c.client == fd, !c.finished {
            c.client = -1
            stopCurrent("stopped by its player")
        } else if let c = current, c.client == fd {
            c.client = -1
        }
    }

    func command(_ line: String) {
        let cmd = line.trimmingCharacters(in: .whitespacesAndNewlines)
        switch cmd {
        case "pause":
            guard let c = current, !c.paused else { return }
            c.offset = position(c)
            c.paused = true
            c.pausedAt = Date()
            token += 1
            player?.stop()
            setPlaying(false)
            event("paused", ["id": c.id, "position": seconds(c, c.offset), "seconds": c.seconds, "path": c.path])
        case "resume":
            guard let c = current, c.paused else { return }
            c.paused = false
            schedule(c)
            event("resumed", ["id": c.id, "position": seconds(c, c.offset)])
        case "stop":
            stopCurrent("stopped")
        case "status":
            var info = deviceInfo()
            info["vp"] = vpActive
            info["running"] = engine?.isRunning ?? false
            info["playing"] = current.map { !$0.paused } ?? false
            event("status", info)
        case "quit":
            stopCurrent("stopped")
            teardown()
            if let s = socketPath { unlink(s) }
            exit(0)
        default:
            break
        }
    }

    // Another app capturing audio (a call, dictation) while voice processing runs
    // could get processed or starved input: step aside, and come back once it is
    // done. Checked every second; the voice is not interrupted mid-playback.
    var othersSince: Date?
    var noOthersSince: Date?
    func checkOthers() {
        guard yieldToOthers, engine != nil else { return }
        if let c = current, !c.finished { return }
        let busy = !otherInputProcesses().isEmpty
        let now = Date()
        if busy { noOthersSince = nil; if othersSince == nil { othersSince = now } }
        else { othersSince = nil; if noOthersSince == nil { noOthersSince = now } }
        if busy && vpActive, let since = othersSince, now.timeIntervalSince(since) >= 1 {
            requestRebuild("another app started using audio input", after: 0)
        } else if !busy && yielding, let since = noOthersSince, now.timeIntervalSince(since) >= 5 {
            requestRebuild("no other app is using audio input now", after: 0)
        }
    }

    // A paused voice must not wait for ever if the listener never decides.
    func watchdog() {
        ctl.asyncAfter(deadline: .now() + 1.0) { [weak self] in
            guard let self = self else { return }
            if let c = self.current, c.paused, Date().timeIntervalSince(c.pausedAt) > 60 {
                self.stopCurrent("paused too long")
            }
            self.checkOthers()
            self.watchdog()
        }
    }

    // ---- start

    func start() {
        signal(SIGPIPE, SIG_IGN)
        // Follow the default devices and any reconfiguration of the engine.
        for selector in [kAudioHardwarePropertyDefaultInputDevice, kAudioHardwarePropertyDefaultOutputDevice] {
            var addr = AudioObjectPropertyAddress(mSelector: selector, mScope: kAudioObjectPropertyScopeGlobal,
                                                  mElement: kAudioObjectPropertyElementMain)
            let what = selector == kAudioHardwarePropertyDefaultInputDevice ? "input" : "output"
            AudioObjectAddPropertyListenerBlock(AudioObjectID(kAudioObjectSystemObject), &addr, ctl) { [weak self] _, _ in
                self?.event_deviceChanged(what)
            }
        }
        // Starting voice processing itself posts one of these while the engine runs
        // on; rebuild only if the engine stopped or the default devices changed.
        NotificationCenter.default.addObserver(forName: .AVAudioEngineConfigurationChange, object: nil,
                                               queue: nil) { [weak self] n in
            guard let self = self else { return }
            let obj = n.object as AnyObject?
            self.ctl.async {
                guard let e = self.engine, obj === e else { return }
                let same = systemProperty(kAudioHardwarePropertyDefaultInputDevice) == self.builtInput
                    && systemProperty(kAudioHardwarePropertyDefaultOutputDevice) == self.builtOutput
                if !same {
                    self.requestRebuild("the default devices changed")
                    return
                }
                if e.isRunning { return }
                // Same devices, engine stopped (voice processing does this once as it
                // settles): start the same engine again, as Apple recommends.
                let playing = self.current.map { !$0.finished && !$0.paused } ?? false
                if let c = self.current, playing { c.offset = self.position(c) }
                self.restarts.append(Date())
                self.restarts = self.restarts.filter { Date().timeIntervalSince($0) < 30 }
                do {
                    if self.restarts.count > 5 { throw NSError(domain: "hal-duplex", code: 5) }
                    try e.start()
                    if let c = self.current, playing { self.schedule(c) }
                } catch {
                    self.requestRebuild("the audio engine stopped")
                }
            }
        }
        ctl.sync { build("start") }
        watchdog()
        if let s = socketPath { serveSocket(s) }
        // Commands from the listener; end of input means it has gone.
        Thread.detachNewThread { [weak self] in
            while let line = readLine(strippingNewline: true) {
                self?.ctl.sync { self?.command(line) }
            }
            self?.ctl.sync {
                self?.stopCurrent("stopped")
                self?.teardown()
            }
            if let s = self?.socketPath { unlink(s) }
            exit(0)
        }
    }

    func event_deviceChanged(_ what: String) {
        event("device", ["changed": what])
        requestRebuild("the default \(what) changed")
    }

    // ---- the socket for `play`

    func serveSocket(_ path: String) {
        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else {
            event("error", ["why": "cannot create the socket"])
            return
        }
        unlink(path)
        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let bytes = Array(path.utf8)
        guard bytes.count < MemoryLayout.size(ofValue: addr.sun_path) else {
            event("error", ["why": "socket path too long"])
            return
        }
        withUnsafeMutableBytes(of: &addr.sun_path) { p in
            for (i, b) in bytes.enumerated() { p[i] = b }
            p[bytes.count] = 0
        }
        let old = umask(0o077)
        let ok = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(fd, $0, socklen_t(MemoryLayout<sockaddr_un>.size)) }
        }
        umask(old)
        guard ok == 0, listen(fd, 8) == 0 else {
            event("error", ["why": "cannot listen on the socket"])
            return
        }
        chmod(path, 0o600)
        Thread.detachNewThread { [weak self] in
            while true {
                let c = accept(fd, nil, nil)
                if c < 0 {
                    if errno == EINTR { continue }
                    return
                }
                Thread.detachNewThread { self?.serveClient(c) }
            }
        }
    }

    func serveClient(_ fd: Int32) {
        var line = Data()
        var byte: UInt8 = 0
        while line.count < 4096 {
            let n = read(fd, &byte, 1)
            if n <= 0 { break }
            if byte == 0x0A { break }
            line.append(byte)
        }
        let text = String(data: line, encoding: .utf8) ?? ""
        if text.hasPrefix("play ") {
            let path = String(text.dropFirst(5))
            let reply = ctl.sync { play(path, client: fd) }
            _ = writeAll(fd, (reply + "\n").data(using: .utf8)!)
            if reply.hasPrefix("started") {
                // Wait for the client to go: when it is killed, its voice stops.
                var sink = [UInt8](repeating: 0, count: 64)
                while read(fd, &sink, sink.count) > 0 {}
                ctl.sync { clientGone(fd) }
            }
        } else if text == "ping" {
            _ = writeAll(fd, "pong\n".data(using: .utf8)!)
        }
        close(fd)
    }
}

// MARK: - play (the client)

func afplay(_ path: String) -> Never {
    let args = ["/usr/bin/afplay", path]
    var cargs = args.map { strdup($0) }
    cargs.append(nil)
    execv("/usr/bin/afplay", &cargs)
    note("cannot run afplay")
    exit(1)
}

func runPlay(_ file: String, socketPath: String) -> Never {
    let path = file.hasPrefix("/") ? file : FileManager.default.currentDirectoryPath + "/" + file
    let fd = socket(AF_UNIX, SOCK_STREAM, 0)
    if fd < 0 { afplay(path) }
    var addr = sockaddr_un()
    addr.sun_family = sa_family_t(AF_UNIX)
    let bytes = Array(socketPath.utf8)
    if bytes.count >= MemoryLayout.size(ofValue: addr.sun_path) { afplay(path) }
    withUnsafeMutableBytes(of: &addr.sun_path) { p in
        for (i, b) in bytes.enumerated() { p[i] = b }
        p[bytes.count] = 0
    }
    let ok = withUnsafePointer(to: &addr) {
        $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { connect(fd, $0, socklen_t(MemoryLayout<sockaddr_un>.size)) }
    }
    if ok != 0 { afplay(path) }   // no helper: play it the ordinary way
    var tv = timeval(tv_sec: 2, tv_usec: 0)
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, socklen_t(MemoryLayout<timeval>.size))
    if !writeAll(fd, "play \(path)\n".data(using: .utf8)!) { afplay(path) }
    func readLineFrom() -> String? {
        var line = Data()
        var byte: UInt8 = 0
        while true {
            let n = read(fd, &byte, 1)
            if n <= 0 { return line.isEmpty ? nil : String(data: line, encoding: .utf8) }
            if byte == 0x0A { return String(data: line, encoding: .utf8) }
            line.append(byte)
        }
    }
    guard let first = readLineFrom(), first.hasPrefix("started") else {
        close(fd)
        afplay(path)          // it could not play it (or did not answer in time)
    }
    let parts = first.split(separator: " ")
    let seconds = parts.count >= 3 ? Double(parts[2]) ?? 600 : 600
    var wait = timeval(tv_sec: Int(seconds) + 30, tv_usec: 0)   // it may be paused for a while
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &wait, socklen_t(MemoryLayout<timeval>.size))
    let done = readLineFrom()
    close(fd)
    exit(done?.hasPrefix("done") == true ? 0 : 1)
}

// MARK: - main

var args = Array(CommandLine.arguments.dropFirst())
func option(_ name: String) -> String? {
    if let i = args.firstIndex(of: name), i + 1 < args.count {
        let v = args[i + 1]
        args.removeSubrange(i...(i + 1))
        return v
    }
    return nil
}
func flag(_ name: String) -> Bool {
    if let i = args.firstIndex(of: name) {
        args.remove(at: i)
        return true
    }
    return false
}

let command = args.first ?? ""
if !args.isEmpty { args.removeFirst() }
switch command {
case "serve":
    let socketPath = option("--socket")
    let mode = option("--mode") ?? "auto"
    let ducking = option("--ducking") ?? "min"
    let agc = flag("--agc")
    let debugLevels = flag("--debug-levels")
    let yieldToOthers = flag("--yield-to-other-input")
    guard ["auto", "vp", "raw"].contains(mode), ["min", "default", "mid", "max"].contains(ducking) else {
        note("usage: hal-duplex serve [--socket PATH] [--mode auto|vp|raw] [--ducking min|default|mid|max] [--agc]")
        exit(2)
    }
    let d = Duplex(socketPath: socketPath, mode: mode, ducking: ducking, agc: agc)
    d.debugLevels = debugLevels
    d.yieldToOthers = yieldToOthers
    d.start()
    RunLoop.main.run()
case "play":
    let socketPath = option("--socket") ?? ""
    guard let file = args.first else {
        note("usage: hal-duplex play FILE [--socket PATH]")
        exit(2)
    }
    if socketPath.isEmpty { afplay(file) }
    runPlay(file, socketPath: socketPath)
case "probe":
    var info = deviceInfo()
    info["other_input"] = otherInputProcesses().map { ["pid": Int($0.0), "bundle": $0.1] }
    if let d = try? JSONSerialization.data(withJSONObject: info, options: [.sortedKeys]) {
        FileHandle.standardOutput.write(d)
        FileHandle.standardOutput.write("\n".data(using: .utf8)!)
    }
    exit(0)
default:
    note("usage: hal-duplex serve|play|probe (see the top of HALDuplex.swift)")
    exit(2)
}
