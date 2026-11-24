"""Command line: `bin/hal <command>`.

    hal on                 unmute: hooks speak Claude's replies
    hal off                mute (and stop anything playing); hooks do nothing
    hal stop               stop speaking now, stay unmuted
    hal status             muted?, speaking?, model present?, config
    hal last               what HAL last said in place of a full reply
    hal say [TEXT]         speak TEXT (or stdin), cleaned and capped like a reply;
                           works while muted. --wait blocks until done,
                           --wav FILE renders to FILE instead of playing,
                           --raw skips cleaning and the length cap (pronunciation
                           fixes such as HAL -> Hal still apply)
    hal clean              print what would be spoken for Markdown on stdin
    hal desk start|stop|status|run|ask TEXT
                           the front desk (see hal/desk.py); used when
                           front_desk is true in config.json

    hal hook               (Claude Code hook entry point; reads event JSON on stdin)
"""

from __future__ import annotations

import argparse
import sys


def _text_arg(words: list[str]) -> str:
    if words and words != ["-"]:
        return " ".join(words)
    if sys.stdin.isatty():
        print("hal: reading text from stdin (Ctrl-D to finish)", file=sys.stderr)
    return sys.stdin.read()


def _say(args) -> int:
    from . import config, speaker
    from .textclean import pronounce, speakable

    text = _text_arg(args.text)
    if args.raw:
        text = pronounce(text)
    else:
        cfg = config.load_config()
        text = speakable(text, cfg["max_chars"], cfg["overflow_message"])
    text = text.strip()
    if not text:
        print("hal: nothing speakable", file=sys.stderr)
        return 1
    if args.wav:
        seconds = speaker.synthesize(text, args.wav, length_scale=config.load_config()["length_scale"])
        print(f"{args.wav}: {seconds:.2f} s")
        return 0
    proc = speaker.speak(text)
    if args.wait and proc is not None:
        try:
            proc.wait()
        except KeyboardInterrupt:
            speaker.stop()
            return 130
    return 0


def _status() -> int:
    from . import config, speaker

    cfg = config.load_config()
    print(f"muted:    {'yes' if config.is_muted() else 'no'}")
    print(f"speaking: {'yes' if speaker.is_speaking() else 'no'}")
    print(f"model:    {config.MODEL} ({'present' if config.MODEL.exists() else 'MISSING: run scripts/fetch-model.sh'})")
    here = "yes" if config.is_voice_session(cfg) else "no (start Claude Code with `hal claude`)"
    print(f"voice:    {cfg['voice_sessions']} sessions; this environment has the voice: {here}")
    print(f"speech:   {cfg['spoken_reply']} of each reply; acknowledge prompts: {'yes' if cfg['acknowledge'] else 'no'}")
    if cfg["front_desk"]:
        from . import desk

        info = desk.status()
        running = f"running (pid {info['pid']})" if info else "not running (starts with the next prompt)"
        print(f"desk:     on, {running}; social remarks {'blocked' if cfg['desk_block'] else 'passed on'}")
    else:
        print("desk:     off")
    print(f"config:   max_chars={cfg['max_chars']} length_scale={cfg['length_scale']} "
          f"speak_in_print_mode={cfg['speak_in_print_mode']}")
    print(f"log:      {config.LOG_FILE}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    # Internal entry points, kept off the help text.
    if argv[:1] == ["hook"]:
        from . import hook
        return hook.main()
    if argv[:1] == ["_worker"] and len(argv) in (2, 3):
        from . import speaker
        return speaker.run_worker(*argv[1:])
    if argv[:1] == ["desk"]:
        from . import desk
        return desk.main(argv[1:])

    parser = argparse.ArgumentParser(prog="hal", description="HAL 9000 voice for Claude Code.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("on", help="unmute: speak Claude's replies")
    sub.add_parser("off", help="mute and stop speaking")
    sub.add_parser("stop", help="stop speaking now")
    sub.add_parser("status", help="show state")
    sub.add_parser("last", help="print what HAL last said in place of a full reply")
    say = sub.add_parser("say", help="speak text (argument or stdin)")
    say.add_argument("text", nargs="*")
    say.add_argument("--wait", action="store_true", help="block until finished (Ctrl-C stops)")
    say.add_argument("--wav", metavar="FILE", help="write a WAV file instead of playing")
    say.add_argument("--raw", action="store_true",
                     help="skip Markdown cleaning and the length cap (pronunciation fixes still apply)")
    sub.add_parser("clean", help="print the speakable form of Markdown read from stdin")
    sub.add_parser("desk", help="the front desk: start|stop|status|run|ask TEXT")
    args = parser.parse_args(argv)

    from . import config, speaker

    if args.command == "on":
        config.set_muted(False)
        print("HAL: on")
    elif args.command == "off":
        config.set_muted(True)
        speaker.stop()
        print("HAL: off (muted)")
    elif args.command == "stop":
        speaker.stop()
    elif args.command == "status":
        return _status()
    elif args.command == "last":
        print(speaker.last_spoken().get("text") or "(the last reply was read in full, or nothing has been spoken)")
    elif args.command == "say":
        return _say(args)
    elif args.command == "clean":
        from .textclean import speakable

        cfg = config.load_config()
        print(speakable(sys.stdin.read(), cfg["max_chars"], cfg["overflow_message"]))
    return 0
