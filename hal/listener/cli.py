"""`hal-listen`: the always-listening front end for the HAL voice.

    hal-listen run [options]   listen in the foreground, showing what is heard and decided
                               (--duplex aec: also while the voice speaks)
    hal-listen start           listen in the background (one listener at a time)
    hal-listen stop            stop the background listener
    hal-listen pause           stop listening (the microphone is closed) until resumed
    hal-listen resume          listen again
    hal-listen status          listener, session and settings
    hal-listen devices         list audio input devices
    hal-listen mic-check       open the microphone for a few seconds; print levels only
    hal-listen type TEXT       type TEXT into the session (same safety checks as speech)
    hal-listen check           may text be typed into the session right now?

    hal-listen pty -- CMD...   run CMD under the pty proxy (what `hal claude` should use)
    hal-listen hook            Claude Code hook entry point (event JSON on stdin)
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import signal
import subprocess
import sys
import time

from . import settings

_LOCK_NAME = "listener.lock"


def _console(enabled: bool = True):
    def emit(msg: str) -> None:
        if enabled:
            print(f"{time.strftime('%H:%M:%S')}  {msg}", flush=True)
    return emit


def _acquire_lock(wait: float = 0.0) -> int | None:
    """The listener lock (one listener at a time). wait > 0 retries briefly, so
    a concurrent `status` probing the lock can't make a starting listener quit."""
    d = settings.listen_dir()
    d.mkdir(parents=True, exist_ok=True)
    fd = os.open(d / _LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o600)
    deadline = time.monotonic() + wait
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except OSError:
            if time.monotonic() >= deadline:
                os.close(fd)
                return None
            time.sleep(0.05)


def _listener_info() -> dict:
    try:
        info = json.loads((settings.listen_dir() / "listener.json").read_text(encoding="utf-8"))
        return info if isinstance(info, dict) else {}
    except (OSError, ValueError):
        return {}


def _running_pid() -> int:
    """PID of the running listener, or 0 (the lock is the source of truth)."""
    fd = _acquire_lock()
    if fd is not None:
        os.close(fd)
        return 0
    pid = _listener_info().get("pid")
    return pid if isinstance(pid, int) and pid > 1 else -1


# ------------------------------------------------------------ the listener itself

def _build(args, cfg: dict, console, proxy: dict | None):
    """Load the models and assemble a Listener (heavy imports happen here)."""
    from . import control, decider, gate, injector, screen, service, vad

    vad_path = settings.project_path(cfg["vad_model"])
    if not vad_path.exists():
        raise SystemExit(f"hal-listen: no VAD model at {vad_path}; run scripts/setup-listen.sh")
    started = time.monotonic()
    console("loading the speech-to-text model ...")
    if getattr(args, "fake_asr", None):
        asr = _FakeASR(args.fake_asr)
    else:
        from .asr import ParakeetASR

        asr = ParakeetASR(cfg["asr_model"])
        asr.warmup()
    console(f"models ready in {time.monotonic() - started:.1f}s")

    clock = time.monotonic
    voice = service.SilentVoice(console) if getattr(args, "silent", False) else service.VoiceOut()
    speaking_gate = gate.SpeakingGate(settings.speaker_pid_files(), cfg["echo_tail"], own_speaking=voice.busy)
    if getattr(args, "file", None):
        from .audio import ArraySource, decode_file

        if args.fast:
            # Faster than real time: timers run on the file's own clock, and with
            # no real-time voice to compete with there is nothing to gate.
            clock = _FileClock()
            speaking_gate = _OpenGate()
        source = ArraySource(decode_file(args.file), realtime=not args.fast, lead_in=0.5, clock=clock)
        source.name = os.path.basename(args.file)
    elif cfg.get("duplex") == "aec":
        # Listen while the voice speaks: the audio helper with echo cancellation, on the
        # system's default devices (it follows changes); half-duplex if it is unavailable.
        from .duplex import DuplexSource

        if getattr(args, "device", None) is not None or cfg["device"] is not None:
            console("duplex \"aec\" uses the system's default input and output; --device / \"device\" is ignored")
        source = DuplexSource(cfg, console=console)
    else:
        from .audio import MicSource

        source = MicSource(args.device if getattr(args, "device", None) is not None else cfg["device"])
    inj = injector.Injector(cfg, client_factory=control.ProxyClient.discover, say=voice, console=console)
    lex = None
    if cfg["spoken_label"] and cfg["lexicon"]:
        from . import lexicon, logs

        try:
            lex = lexicon.load(cfg, log=logs.log)
            console(f"lexicon: {lex.describe()}")
            logs.log(f"lexicon: {lex.describe()}")
        except Exception as exc:  # no corrections rather than no listener
            logs.log(f"lexicon: could not load ({exc!r}); no corrections")
            console(f"lexicon: could not load ({exc!r}); no corrections")
    voice_check = None
    if cfg["voice_check"] != "off":
        from . import logs, voiceprint

        voice_check = voiceprint.VoiceCheck(cfg, log=logs.log)
        if voice_check.ready():
            if voice_check.embedder() is not None:
                logs.log(f"voice check: {cfg['voice_check']}; threshold {cfg['voice_threshold']:g}")
        elif not voiceprint.model_path(cfg).exists():
            voice_check = None   # no model: nothing to check with (scripts/fetch-voice-model.sh)
    turn_judge = None
    if cfg["turn_taking"] == "semantic" and cfg["turn_judge"]:
        from . import judge

        turn_judge = judge.DeskJudge(cfg["turn_judge_timeout"], on_absent=judge.wake_desk)
    return service.Listener(cfg, source=source, vad=vad.SileroVAD(str(vad_path)), asr=asr, gate=speaking_gate,
                            injector=inj, decider=decider.Decider(cfg), voice=voice, console=console, clock=clock,
                            proxy_pid=proxy["pid"] if proxy else None,
                            proxy_socket=proxy.get("socket") if proxy else None,
                            dry_run=args.dry_run, show_ignored=args.show_ignored,
                            linger=getattr(args, "linger", 0.0), judge=turn_judge, lexicon=lex,
                            screen_locked=screen.locked if (cfg["pause_when_locked"] and not args.dry_run
                                                            and not getattr(args, "file", None)) else None,
                            voice_check=voice_check)


class _FileClock:
    """`run --file F --fast`: time is the position in the file."""

    def __init__(self):
        self.t = time.monotonic()

    def __call__(self) -> float:
        return self.t

    def advance_to(self, t: float) -> None:
        self.t = max(self.t, t)


class _OpenGate:
    reply_end = -1e9

    def blocked(self, t) -> bool:  # noqa: ARG002
        return False


class _FakeASR:
    """--fake-asr FILE: answers from a list instead of a model (pipeline tests only)."""

    name = "fake"

    def __init__(self, path: str):
        with open(path, encoding="utf-8") as f:
            self.answers = [line.rstrip("\n") for line in f]

    def transcribe(self, audio) -> str:  # noqa: ARG002
        return self.answers.pop(0) if self.answers else ""


def _listen(args, background: bool) -> int:
    cfg = settings.load()
    if getattr(args, "duplex", None):
        cfg["duplex"] = args.duplex
    console = _console(not background)
    lock = _acquire_lock(wait=1.0)
    if lock is None:
        print(f"hal-listen: a listener is already running (pid {_listener_info().get('pid', '?')})", file=sys.stderr)
        return 1
    status_file = settings.listen_dir() / "listener.json"
    status_file.write_text(json.dumps({"pid": os.getpid(), "state": "loading", "started": time.time()}))
    from . import control, logs

    proxy = control.find_proxy()
    if proxy is None and not args.dry_run:
        msg = "no Claude Code session is running under the pty proxy (start it with `hal claude`, or use --dry-run)"
        logs.log(f"not started: {msg}")
        print(f"hal-listen: {msg}", file=sys.stderr)
        return 1
    proxy_pid = proxy["pid"] if proxy else None
    listener = _build(args, cfg, console, proxy)
    signal.signal(signal.SIGTERM, listener.request_stop)
    signal.signal(signal.SIGHUP, listener.request_stop)
    logs.log(f"started ({'background' if background else 'foreground'}; "
             f"{'dry run' if args.dry_run else f'serving proxy {proxy_pid}'}; source {listener.source.name})")
    try:
        listener.run()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        logs.log(f"stopped by an error: {exc!r}")
        raise
    finally:
        logs.log(f"stopped ({listener.counts.__dict__})")
        os.close(lock)
    if args.report:  # for the end-to-end test: decisions, latencies, what was said (never ignored words)
        t0 = getattr(listener.source, "t0", None)
        t0_audio = listener.source.time_of(0) if hasattr(listener.source, "time_of") and t0 is not None else None
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump({"events": listener.events, "submitted": listener.injector.submitted,
                       "said": getattr(listener.voice, "said", []), "counts": listener.counts.__dict__,
                       "audio_t0": t0_audio,
                       "audio_t0_wall": None if t0_audio is None else time.time() - (time.monotonic() - t0_audio)},
                      f)
    return 0


def _start(args) -> int:
    pid = _running_pid()
    if pid:
        if not args.quiet:
            print(f"hal-listen: already running (pid {pid if pid > 0 else '?'})")
        return 0
    from . import control

    if control.find_proxy() is None:
        if not args.quiet:
            print("hal-listen: no Claude Code session is running under the pty proxy (start it with `hal claude`)",
                  file=sys.stderr)
        return 1
    settings.listen_dir().mkdir(parents=True, exist_ok=True)
    boot = "import sys; sys.path.insert(0, sys.argv[1]); from hal.listener.cli import main; sys.exit(main(sys.argv[2:]))"
    with open(settings.log_file(), "ab") as log:
        proc = subprocess.Popen([sys.executable, "-E", "-P", "-c", boot, str(settings.ROOT), "_serve"],
                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=log,
                                start_new_session=True, cwd=str(settings.ROOT))
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            if not args.quiet:
                print(f"hal-listen: the listener exited at once (code {proc.returncode}); see {settings.log_file()}",
                      file=sys.stderr)
            return 1
        if _listener_info().get("pid") == proc.pid or _running_pid() == proc.pid:
            break
        time.sleep(0.1)
    if not args.quiet:
        print(f"hal-listen: listening in the background (pid {proc.pid}); log: {settings.log_file()}")
    return 0


def _stop(args) -> int:  # noqa: ARG001
    pid = _running_pid()
    if not pid:
        print("hal-listen: not running")
        return 0
    if pid < 0:
        print("hal-listen: a listener holds the lock but its pid is unknown", file=sys.stderr)
        return 1
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and _running_pid():
        time.sleep(0.1)
    if _running_pid():
        print(f"hal-listen: pid {pid} did not stop", file=sys.stderr)
        return 1
    print("hal-listen: stopped")
    return 0


def _pause(on: bool) -> int:
    flag = settings.listen_dir() / "paused"
    if on:
        flag.parent.mkdir(parents=True, exist_ok=True)
        flag.touch()
        print("hal-listen: paused (the microphone closes within a second)")
    else:
        flag.unlink(missing_ok=True)
        print("hal-listen: listening" if _running_pid() else "hal-listen: resumed (but no listener is running)")
    return 0


def _status(args) -> int:  # noqa: ARG001
    from . import control

    cfg = settings.load()
    pid = _running_pid()
    info = _listener_info()
    paused = (settings.listen_dir() / "paused").exists()
    if pid:
        counts = info.get("counts", {})
        why = " (paused)" if paused else (" (screen locked)" if info.get("screen_locked") else "")
        print(f"listener: running (pid {pid}), {info.get('state', '?')}{why}; "
              f"source {info.get('source', '?')}; submitted {counts.get('submitted', 0)}, "
              f"ignored {counts.get('ignored', 0)}, discarded while speaking {counts.get('discarded', 0)}")
    else:
        print(f"listener: not running{' (paused flag set)' if paused else ''}")
    proxy = control.find_proxy()
    if proxy:
        try:
            st = control.ProxyClient(proxy["socket"], timeout=2).status()
            dialogs = ", ".join(st.get("dialogs") or []) or "none"
            print(f"session:  {proxy.get('argv0')} pid {st.get('child_pid')} under proxy {proxy['pid']}; "
                  f"turn {st.get('turn')}; dialogs {dialogs}; "
                  f"can type: {'yes' if st.get('can_type') else 'no (' + str(st.get('reason')) + ')'}")
        except control.Unreachable as exc:
            print(f"session:  proxy {proxy['pid']} registered but not answering ({exc})")
    else:
        print("session:  none under the pty proxy (start Claude Code with `hal claude`)")
    asr = settings.project_path(cfg["asr_model"])
    vad = settings.project_path(cfg["vad_model"])
    print(f"models:   {asr.name} ({'present' if asr.exists() else 'missing'}), "
          f"{vad.name} ({'present' if vad.exists() else 'missing'})")
    print(f"names:    {', '.join(cfg['names'])}; when called: {', '.join(cfg['vocative_names'])}")
    print(f"timing:   end of utterance after {cfg['end_silence']}s silence; voice tail {cfg['echo_tail']}s; "
          f"follow-up window {cfg['follow_up_seconds'] or 'off'}; armed {cfg['armed_seconds']}s; "
          f"during a spoken request's turn: {'no name needed' if cfg['during_turn'] else 'name needed'}")
    if cfg["turn_taking"] == "semantic":
        from . import judge

        judged = "off"
        if cfg["turn_judge"]:
            judged = "front desk" + ("" if os.path.exists(judge.socket_path()) else " (not running now: timer instead)")
        held = (info.get("turn") or {}) if pid else {}
        now = f"; holding {held['fragments']} fragment(s) ({held['verdict']})" if held else ""
        print(f"turns:    by what was said; finished: sent after {cfg['turn_end_silence']:g}s of silence; "
              f"unfinished: up to {cfg['turn_max_wait']:g}s; unclear: judge {judged}, else "
              f"{cfg['turn_unclear_wait']:g}s{now}")
    else:
        print("turns:    by silence (every utterance is a request)")
    duplex = (info.get("duplex") or {}) if pid else {}
    if cfg["duplex"] == "aec":
        now = ""
        if duplex:
            now = f"; now {duplex.get('mode')}"
            if duplex.get("input"):
                now += f" (input {duplex.get('input')}, output {duplex.get('output')}, {duplex.get('output_kind')})"
            if duplex.get("why"):
                now += f": {duplex['why']}"
        print(f"duplex:   aec: listens while the voice speaks; speaking over it "
              f"{'interrupts it' if cfg['barge_in'] else 'does not interrupt it'}{now}")
    else:
        print("duplex:   half: nothing is heard while the voice speaks")
    if cfg["spoken_label"]:
        marks = [f"lexicon ({settings.project_path(cfg['lexicon_file']).name} and "
                 f"{len(cfg['lexicon_dirs'])} folders)" if cfg["lexicon"] else "",
                 f"doubtful words below {cfg['doubt_threshold']:g}" if cfg["doubt_marks"] else "",
                 f"pauses of {cfg['pause_mark_seconds']:g}s or more" if cfg["pause_marks"] else "",
                 "interruptions" if cfg["interruption_note"] else ""]
        marks = [m for m in marks if m]
        print(f"typed:    labelled [spoken HH:MM]; marks: {', '.join(marks) or 'none'}")
    else:
        print("typed:    as heard, without a label (spoken_label is false)")
    print(f"voice:    {_voice_summary(cfg)}")
    print(f"log:      {settings.log_file()}")
    return 0


def _voice_summary(cfg: dict) -> str:
    from . import voiceprint

    if cfg["voice_check"] == "off":
        return "not checked (voice_check is off)"
    vp = voiceprint.load()
    if vp is None:
        return f"{cfg['voice_check']}, but no voiceprint yet (hal-listen enrol)"
    if not voiceprint.model_path(cfg).exists():
        return f"{cfg['voice_check']}, but the model is missing (scripts/fetch-voice-model.sh)"
    what = {"log": "scores logged, nothing refused", "enforce": "other voices ignored"}[cfg["voice_check"]]
    return (f"{cfg['voice_check']}: {what}; threshold {cfg['voice_threshold']:g}; "
            f"voiceprint of {len(vp.embeddings)} sentences ({vp.updated})")


def _enrol(args) -> int:
    """Record you reading sentences and keep their embeddings as your voiceprint."""
    from . import voiceprint
    from .audio import decode_file

    cfg = settings.load()
    path = voiceprint.model_path(cfg)
    if not path.exists():
        print(f"hal-listen enrol: no model at {path}; run scripts/fetch-voice-model.sh", file=sys.stderr)
        return 1
    embed = voiceprint.Embedder(path)
    old = voiceprint.load() if args.add else None
    if args.add and old is None:
        print("no voiceprint yet: enrolling afresh")
    if args.file:
        embeddings = []
        for f in args.file:
            e = embed(decode_file(f))
            print(f"  {os.path.basename(f)}: {'ok' if e is not None else 'too short, skipped'}")
            if e is not None:
                embeddings.append(e)
    else:
        embeddings = _enrol_live(args, cfg, embed)
        if embeddings is None:
            return 1
    if len(embeddings) < (1 if old is not None else args.minimum):
        print(f"hal-listen enrol: only {len(embeddings)} usable sentence(s); nothing saved", file=sys.stderr)
        return 1
    vp = voiceprint.new_voiceprint(embed.name, embeddings, old)
    saved = voiceprint.save(vp)
    print(f"voiceprint: {len(vp.embeddings)} sentences, saved to {saved} (readable by you only; no audio kept)")
    scores = vp.self_scores()
    if scores:
        import numpy as np

        print(f"consistency: each sentence against the others: lowest {min(scores):.2f}, "
              f"median {float(np.median(scores)):.2f}")
    print(f"voice check: {_voice_summary(cfg)}")
    return 0


def _enrol_live(args, cfg: dict, embed) -> list | None:
    """Show each sentence, wait for it to be read, embed it. Audio stays in memory."""
    from . import voiceprint
    from .audio import MicSource
    from .segmenter import Segmenter, Utterance
    from .vad import SileroVAD

    sentences = voiceprint.SENTENCES[: max(1, args.count)]
    flag = settings.listen_dir() / "paused"
    paused_here = bool(_running_pid()) and not flag.exists()
    if paused_here:
        _pause(True)
        time.sleep(1.2)
    src = MicSource(args.device if args.device is not None else cfg["device"])
    vad = SileroVAD(str(settings.project_path(cfg["vad_model"])))
    seg = Segmenter(cfg)
    embeddings = []
    print(f"Read each sentence aloud in your normal voice, at your usual distance ({len(sentences)} in all).")
    print("Ctrl-C stops without saving anything.\n")
    try:
        src.start()
        for k, sentence in enumerate(sentences, 1):
            print(f"[{k}/{len(sentences)}] {sentence}")
            deadline = time.monotonic() + 30
            while True:
                if time.monotonic() > deadline:
                    print("    nothing heard; skipped")
                    break
                item = src.read(0.2)
                if item is None:
                    continue
                t, frame = item
                res = seg.push(frame, vad(frame), t)
                if not isinstance(res, Utterance) or res.audio is None:
                    continue
                if res.speech_seconds < 1.0:
                    print("    too short; please read the whole sentence")
                    continue
                e = embed(voiceprint.speech_only(res.audio, res.start, res.speech_start, res.speech_end))
                res.audio = None
                if e is None:
                    print("    couldn't use that; please read it again")
                    continue
                embeddings.append(e)
                print(f"    ok ({res.speech_seconds:.1f}s)")
                break
    except KeyboardInterrupt:
        print("\nstopped; nothing saved")
        return None
    finally:
        src.stop()
        if paused_here:
            _pause(False)
    return embeddings


def _voiceprint(args) -> int:
    from . import voiceprint
    from .audio import decode_file

    cfg = settings.load()
    if args.action == "forget":
        print("voiceprint: deleted" if voiceprint.forget() else "voiceprint: there was none")
        return 0
    if args.action == "report":
        return _voice_report(cfg)
    vp = voiceprint.load()
    if vp is None:
        print("voiceprint: none (hal-listen enrol makes one)")
        return 1 if args.action == "score" else 0
    if args.action == "score":
        embed = voiceprint.Embedder(voiceprint.model_path(cfg))
        for f in args.files:
            audio = decode_file(f)
            e = embed(audio)
            if e is None:
                print(f"{f}: too short")
                continue
            score = vp.score(e)
            print(f"{f}: {score:.2f} ({'you' if score >= cfg['voice_threshold'] else 'another voice'})")
        return 0
    scores = vp.self_scores()
    print(f"voiceprint: {len(vp.embeddings)} sentences, made {vp.created}, last added to {vp.updated}; "
          f"model {vp.model}")
    if scores:
        print(f"consistency: lowest {min(scores):.2f}, median {sorted(scores)[len(scores) // 2]:.2f}")
    print(f"voice check: {_voice_summary(cfg)}")
    return 0


def _voice_report(cfg: dict) -> int:
    """The scores the listener has logged, to choose voice_threshold."""
    from . import voiceprint

    log = settings.log_file()
    lines = []
    for path in (log.with_name(log.name + ".1"), log):
        with contextlib.suppress(OSError):
            lines += path.read_text(encoding="utf-8", errors="replace").splitlines()
    groups = voiceprint.logged_scores(lines)
    t = cfg["voice_threshold"]
    labels = {"addressed": "requests heard (mostly you)", "unaddressed": "speech not addressed to Claude",
              "refused": "ignored as another voice"}
    if not any(groups.values()):
        print("no voice scores logged yet: with a voiceprint and voice_check \"log\", each utterance's score is logged")
        return 0
    for key, scores in groups.items():
        if not scores:
            continue
        s = sorted(scores)
        q = lambda f: s[min(len(s) - 1, int(f * len(s)))]  # noqa: E731
        below = sum(x < t for x in s)
        print(f"{labels[key]}: {len(s)}; lowest {s[0]:.2f}, 10% {q(0.1):.2f}, median {q(0.5):.2f}, "
              f"highest {s[-1]:.2f}; {below} below the threshold {t:g}")
    print("Requests of yours below the threshold would be refused under \"enforce\"; "
          "set voice_threshold between your scores and other voices'.")
    return 0


def _devices(args) -> int:  # noqa: ARG001
    import sounddevice as sd

    default = sd.default.device[0]
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            mark = "*" if i == default else " "
            print(f"{mark} {i:2d}  {dev['name']}  ({dev['max_input_channels']} ch, {dev['default_samplerate']:.0f} Hz)")
    return 0


def _mic_check(args) -> int:
    """Open the microphone briefly and report levels only: nothing is kept or transcribed."""
    import numpy as np

    from .audio import MicSource
    from .vad import SileroVAD

    cfg = settings.load()
    src = MicSource(args.device if args.device is not None else cfg["device"])
    vad = SileroVAD(str(settings.project_path(cfg["vad_model"])))
    rms, probs = [], []
    src.start()
    try:
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            item = src.read(0.2)
            if item is None:
                continue
            frame = item[1]
            rms.append(float(np.sqrt(np.mean(frame * frame))))
            probs.append(vad(frame))
            del frame
    finally:
        src.stop()
    if not rms:
        print("mic-check: no audio arrived", file=sys.stderr)
        return 1
    db = [20 * np.log10(max(r, 1e-9)) for r in rms]
    speech = sum(p >= cfg["vad_threshold"] for p in probs) * 512 / 16000
    print(f"device:   {src.name}")
    print(f"captured: {len(rms) * 512 / 16000:.1f}s; dropped frames {src.dropped}; stream warnings {src.status_flags}")
    print(f"level:    median {np.median(db):.1f} dBFS, loudest frame {max(db):.1f} dBFS")
    print(f"speech:   {speech:.1f}s scored as speech by the VAD")
    if max(rms) == 0:
        print("all samples are zero: microphone permission is probably denied for this terminal app")
        return 1
    return 0


def _type(args) -> int:
    from . import control

    try:
        reply = control.ProxyClient.discover().type(" ".join(args.text), submit=not args.no_enter)
    except control.Unreachable as exc:
        print(f"hal-listen: {exc}", file=sys.stderr)
        return 1
    if reply.get("ok"):
        return 0
    print(f"hal-listen: not typed: {reply.get('reason')} {reply.get('detail') or ''}".rstrip(), file=sys.stderr)
    return 1


def _check(args) -> int:  # noqa: ARG001
    from . import control

    try:
        reply = control.ProxyClient.discover().check()
    except control.Unreachable as exc:
        print(f"hal-listen: {exc}", file=sys.stderr)
        return 1
    print("can type" if reply.get("ok") else f"cannot type: {reply.get('reason')} {reply.get('detail') or ''}")
    return 0 if reply.get("ok") else 1


def _serve_supervised(ns) -> int:
    """Run the background listener; if it dies of an error, say so and start again.

    The restart replaces this process with a fresh copy of itself (same pid, so
    its locks and files are released and taken again). At most five restarts in
    a row; a run of ten minutes resets the count. Not used with a test file or
    a stand-in transcriber, where an error should simply surface.
    """
    started = time.time()
    try:
        return _listen(ns, background=True)
    except Exception as exc:  # noqa: BLE001 - anything that kills the listener
        from . import logs

        if ns.file or ns.fake_asr or os.environ.get("HAL_LISTEN_NO_RESTART"):
            raise
        count = 0 if time.time() - started > 600 else int(os.environ.get("HAL_LISTEN_RESTARTS", "0") or 0)
        if count >= 5 or not getattr(sys, "orig_argv", None):
            logs.log(f"listener: giving up after {count} restarts in a row ({exc!r})")
            _say_aside("I have stopped listening, because the listener keeps failing. Please check on me at the keyboard.")
            raise
        logs.log(f"listener: restarting after an error ({exc!r}); restart {count + 1} of 5")
        _say_aside("I stopped listening for a moment. I am listening again.")
        os.environ["HAL_LISTEN_RESTARTS"] = str(count + 1)
        time.sleep(2.0)
        os.execv(sys.orig_argv[0], sys.orig_argv)
    return 0


def _say_aside(text: str) -> None:
    """Speak a short notice through the voice, without waiting for it."""
    try:
        subprocess.Popen([str(settings.ROOT / "bin" / "hal"), "say", "--raw", text],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
    except OSError:
        pass


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["hook"]:
        from . import hookstate

        return hookstate.main()
    if argv[:1] == ["pty"]:
        from . import ptyproxy

        return ptyproxy.main(argv[1:])
    if argv[:1] == ["_serve"]:
        # Testing aids (LISTENER.md): HAL_LISTEN_FILE replaces the microphone with an
        # audio file, HAL_LISTEN_FAKE_ASR the model with a list of transcripts.
        ns = argparse.Namespace(dry_run=False, show_ignored=False, file=os.environ.get("HAL_LISTEN_FILE") or None,
                                fast=False, silent=False, device=None, linger=0.0, report=None, duplex=None,
                                fake_asr=os.environ.get("HAL_LISTEN_FAKE_ASR") or None)
        return _serve_supervised(ns)

    parser = argparse.ArgumentParser(prog="hal-listen", description="Always-listening front end for the HAL voice.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="listen in the foreground, showing what is heard and decided")
    run.add_argument("--dry-run", action="store_true", help="decide but don't type (no session needed)")
    run.add_argument("--show-ignored", dest="show_ignored", action="store_true", default=None,
                     help="print the words of ignored utterances (default: only when stdout is a terminal)")
    run.add_argument("--hide-ignored", dest="show_ignored", action="store_false")
    run.add_argument("--file", help="listen to an audio file instead of the microphone (testing)")
    run.add_argument("--fast", action="store_true", help="with --file: don't pace it in real time")
    run.add_argument("--silent", action="store_true", help="print what the voice would say instead of saying it")
    run.add_argument("--device", help="input device (index or name substring)")
    run.add_argument("--duplex", choices=["half", "aec"],
                     help="override the duplex setting: aec listens while the voice speaks (LISTENER.md)")
    run.add_argument("--linger", type=float, default=5.0, help="with --file: seconds to wait for held requests")
    run.add_argument("--report", help=argparse.SUPPRESS)
    run.add_argument("--fake-asr", dest="fake_asr", help=argparse.SUPPRESS)
    start = sub.add_parser("start", help="listen in the background")
    start.add_argument("--quiet", action="store_true")
    sub.add_parser("stop", help="stop the background listener")
    sub.add_parser("pause", help="stop listening until resumed")
    sub.add_parser("resume", help="listen again")
    sub.add_parser("status", help="show the listener, the session and settings")
    sub.add_parser("devices", help="list audio input devices")
    mic = sub.add_parser("mic-check", help="open the microphone briefly and print levels only")
    mic.add_argument("--seconds", type=float, default=3.0)
    mic.add_argument("--device")
    typ = sub.add_parser("type", help="type TEXT into the session and press Enter")
    typ.add_argument("text", nargs="+")
    typ.add_argument("--no-enter", action="store_true", help="type without pressing Enter")
    sub.add_parser("check", help="may text be typed into the session now?")
    enrol = sub.add_parser("enrol", help="record your voice reading sentences, for the voice check")
    enrol.add_argument("--count", type=int, default=20, help="how many sentences to read (default 20)")
    enrol.add_argument("--add", action="store_true", help="add to the voiceprint instead of replacing it")
    enrol.add_argument("--device", help="input device (index or name substring)")
    enrol.add_argument("--file", nargs="+", help="enrol from audio files instead of the microphone")
    enrol.add_argument("--minimum", type=int, default=5, help=argparse.SUPPRESS)
    vpp = sub.add_parser("voiceprint", help="show, test or delete your voiceprint")
    vpp.add_argument("action", nargs="?", default="status", choices=["status", "score", "report", "forget"])
    vpp.add_argument("files", nargs="*", help="with score: audio files to score against the voiceprint")
    args = parser.parse_args(argv)

    if getattr(args, "device", None) is not None and str(args.device).isdigit():
        args.device = int(args.device)
    if args.command == "run":
        if args.show_ignored is None:
            args.show_ignored = sys.stdout.isatty()
        return _listen(args, background=False)
    return {"start": _start, "stop": _stop, "status": _status, "devices": _devices, "mic-check": _mic_check,
            "type": _type, "check": _check, "enrol": _enrol, "voiceprint": _voiceprint,
            "pause": lambda a: _pause(True), "resume": lambda a: _pause(False)}[args.command](args)
