"""Stand-in for an interactive program (claude) under the pty proxy, for the tests.

Puts its terminal in raw mode, like Claude Code, and reports on stdout:
    READY rows=R cols=C        at start
    WINCH rows=R cols=C        on each window-size change
    KEYS <hex>                 each chunk read from the terminal (STANDIN_HEX=1)
    LINE <json>                each line submitted with Enter (\r); Ctrl-U clears the line
Each submitted line is also appended, with a timestamp, to $STANDIN_OUT (if set),
and the command $STANDIN_ON_LINE (if set) is run with the line as an extra argument; the
end-to-end test uses that to fire the HAL hooks the way Claude Code would.

Lines that are commands:
    exit N      exit with status N
    die SIG     kill itself with signal SIG (a number)
    tstp        send itself SIGTSTP (as a program implementing Ctrl-Z would)
    stop        send itself SIGSTOP
"""

import json
import os
import shlex
import signal
import subprocess
import sys
import termios
import time
import tty
import fcntl
import struct


def size():
    rows, cols, _, _ = struct.unpack("HHHH", fcntl.ioctl(0, termios.TIOCGWINSZ, b"\0" * 8))
    return rows, cols


def say(msg):
    os.write(1, (msg + "\r\n").encode())


def on_winch(signum, frame):
    rows, cols = size()
    say(f"WINCH rows={rows} cols={cols}")


def on_cont(signum, frame):
    say("CONT")


def main():
    attrs = termios.tcgetattr(0)
    tty.setraw(0)
    signal.signal(signal.SIGWINCH, on_winch)
    signal.signal(signal.SIGCONT, on_cont)
    rows, cols = size()
    say(f"READY rows={rows} cols={cols} pid={os.getpid()}")
    out = os.environ.get("STANDIN_OUT")
    hook = os.environ.get("STANDIN_ON_LINE")
    show_hex = os.environ.get("STANDIN_HEX") == "1"
    line = b""
    try:
        while True:
            try:
                data = os.read(0, 4096)
            except InterruptedError:
                continue
            if not data:
                return 0
            if show_hex:
                say(f"KEYS {data.hex()}")
            for b in data:
                if b == 0x15:      # Ctrl-U clears the line, as in Claude Code's prompt box
                    line = b""
                    continue
                if b != 0x0D:
                    line += bytes([b])
                    continue
                text = line.decode("utf-8", "replace")
                line = b""
                say(f"LINE {json.dumps(text)}")
                if out:
                    with open(out, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"t": time.time(), "line": text}) + "\n")
                words = text.split()
                if words[:1] == ["exit"] and len(words) == 2:
                    return int(words[1])
                if words[:1] == ["die"] and len(words) == 2:
                    termios.tcsetattr(0, termios.TCSANOW, attrs)
                    os.kill(os.getpid(), int(words[1]))
                    time.sleep(1)
                if words == ["tstp"]:
                    os.kill(os.getpid(), signal.SIGTSTP)
                if words == ["stop"]:
                    os.kill(os.getpid(), signal.SIGSTOP)
                if hook:
                    subprocess.Popen(shlex.split(hook) + [text], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
    finally:
        termios.tcsetattr(0, termios.TCSANOW, attrs)


if __name__ == "__main__":
    sys.exit(main())
