## Starting at login

Instead of typing `bin/hal claude --continue` each day, the voice session can start by itself at login and stay up. Three pieces do this:

- **At login**, a LaunchAgent opens an iTerm2 window in this folder that runs the voice session. Without iTerm2 it uses Terminal.
- **In that window**, `bin/hal-session` runs `bin/hal claude` and keeps it running:
  - **Crashes are restarted.** If Claude Code exits unexpectedly, it starts again after 5 s. If it keeps crashing, the waits grow (10, 20, 40, 60 s), and after five restarts in a row it gives up. A run of ten minutes or more resets the count.
  - **Deliberate exits are respected.** If you quit (`/exit`, Ctrl-C twice, Ctrl-D) or close the window, it stops. It prints one line saying how to start again and leaves a shell open in the project folder.
  - **One conversation a day.** It continues the previous conversation (`--continue`) if the last voice session started the same day, and starts a fresh one on the first start of a new day. The day changes at 04:00, so a restart after midnight still continues the evening's conversation.
  - **Only one session.** If a voice session is already running, however it was started, it does nothing.
- **A menu-bar item** shows whether the session is running, listening, paused or muted, and has the everyday controls.

### Turning it on and off

```sh
scripts/install-autostart.sh --dry-run   # show what it would do
scripts/install-autostart.sh             # install and load both agents
scripts/uninstall-autostart.sh           # unload and remove them (also takes --dry-run)
```

The install script does three things:

- It builds the menu-bar item if needed.
- It copies `launchd/local.hal.voice-session.plist` and `launchd/local.hal.menubar.plist` into `~/Library/LaunchAgents` and loads them.
- It prints each step.

You can run it again safely: it leaves an agent that is already installed and loaded alone, and replaces and reloads one whose file has changed. Loading runs the voice-session agent once straight away, but it opens a window only if no voice session is running. `--no-menubar` leaves the menu-bar item out.

Uninstalling quits the menu-bar item. A voice session that is running keeps running, because it belongs to its window and not to launchd.

### The menu-bar item

The icon is a template image, so the menu bar tints it for light and dark:

| Icon | Meaning |
| --- | --- |
| microphone | The voice session is running and listening. |
| microphone with a slash | Listening is paused. |
| microphone with a cross | The session is running, but no listener is (or the listener has stopped responding). |
| hourglass | The listener is loading its speech model. |
| circular arrows | Claude Code crashed and `bin/hal-session` is waiting to restart it. |
| moon | No voice session is running. |
| speaker with a slash, added on the right | The voice is muted. |

The menu shows three status lines (session, listening, voice), followed by these items:

- **Pause / Resume Listening** runs `hal-listen pause` / `resume`.
- **Mute / Unmute Voice** runs `hal off` / `on`.
- **Stop Speaking Now** runs `hal stop`.
- **Open Decisions Page** is shown only if something answers on `127.0.0.1:8765`.
- **Start Voice Session** is shown only when no session is running. It opens the window as at login.
- **Quit HAL Menu**.

The item reads the state files every 2 seconds (and when the menu opens) and starts no processes for that. It needs no Accessibility or other permission. If you quit it, it stays off until the next login; `launchctl kickstart gui/$(id -u)/local.hal.menubar` brings it back. If it crashes, launchd restarts it.

To check it without the menu bar:

- `menubar/HALMenu --status` prints what it reads from the state files.
- `menubar/HALMenu --print-menu` prints the menu with each item's command.
- `menubar/HALMenu --dry-run` runs it with menu actions printed instead of run.

### Commands and files

| | |
| --- | --- |
| `bin/hal-session` | Run the supervised voice session in this terminal. `--fresh` or `--continue` overrides the day rule; arguments after `--` go to Claude Code. |
| `bin/hal-session --check` | Say whether a voice session is running (exit status 0 if so). |
| `bin/hal-session --dry-run` | Say what it would run, and why. |
| `scripts/open-voice-window.sh` | Open the voice-session window now (what the agent and the menu run). Takes `--dry-run`. |
| `launchd/hal-voice.command` | What the window runs: `bin/hal-session`, then a shell. |
| `menubar/HALMenu.swift`, `menubar/build.sh` | The menu-bar item and its build (`swiftc`, AppKit only). |
| `state/autostart/` | `session.json` (the launcher's state), `last-start.json` (for the day rule), `autostart.log`. |
| `state/launchd-*.log` | Output of the two agents. |

`tests/test_autostart.py` tests the launcher, the window script, the install scripts, the plists and the menu-bar item's state detection. It uses stand-ins for Claude Code, `open` and `launchctl`, a temporary state folder, and a temporary folder in place of `~/Library/LaunchAgents`.

### Known limits

- **iTerm2 asks before running the file.** When asked to open a script file, iTerm2 3.7 asks to confirm: "Run “…/launchd/hal-voice.command”?" Until you answer, no session starts, and nothing here can answer for you. The confirmation has a remembered-choice setting (`NoSyncConfirmRunOpenFile`), but whether the dialog offers "don't ask again" hasn't been checked. Terminal doesn't ask.
- **Why `open` rather than AppleScript.** The window is opened with `open -b com.googlecode.iterm2`. Driving iTerm2 with AppleScript would need macOS's Automation permission for the agent, which is also a one-time prompt, plus a script that can't be tested without opening windows.
- **The network at login.** The agent waits up to 60 s for a network route, then opens the window anyway. Without a connection, Claude Code starts but its requests fail until you are online. The front desk restarts its model if that dies. A network that is up but has no internet (a captive portal, for example) isn't detected.
- **iTerm2 starting at login** may also open or restore its usual windows next to the voice-session window.
- **The microphone permission belongs to the terminal app.** iTerm2 already has it. If the fallback uses Terminal, macOS asks for Terminal's microphone access the first time, and the listener hears silence until you allow it.
- **`--continue` continues the latest conversation in this folder.** That may not be the voice session's. If you used a plain `claude` here in between, a same-day restart continues that conversation instead. The day boundary is `NEW_DAY_HOUR` in `scripts/hal_session.py`.
- **Sessions started by hand.** A session started with `bin/hal claude` is detected, so a second one isn't started. However, it isn't restarted if it crashes, and its start doesn't count for the day rule. A session started with `HAL_LISTEN=0` (no pty proxy) isn't detected at all.
- **What counts as a crash.** Every exit other than 0 (or the conventional 129, 130 and 143, or death by SIGHUP, SIGINT or SIGTERM) counts as a crash. If Claude Code ever exits with another code on purpose, it would be restarted, at most five times.
- **SIGKILL is reported as exit 127.** If Claude Code is killed with SIGKILL, the pty proxy reports exit code 127 and prints a misleading "cannot run". The session is still restarted. The cause is in `_session_leader()` in `hal/listener/ptyproxy.py`: `signal.signal(SIGKILL, SIG_DFL)` raises an error, so that call should be skipped for SIGKILL and SIGSTOP. That fix is not made here.
- **Ctrl-Z suspends the whole window's job.** The shell prompt appears, and `fg` resumes the session.
- **The paths are absolute.** The plists in `launchd/` are templates: the install script replaces `@HAL_ROOT@` with the project's folder in the copies it installs. If the project moves, run the install script again. If the project moves under Documents or Desktop, macOS would also ask before letting the agents read it.
- **macOS lists the agents.** When the agents are installed, macOS shows a "Background Items Added" notice. They are then listed under System Settings > General > Login Items & Extensions, where they can be switched off.
- **Not yet seen for real:** the actual login, the iTerm2 window it opens, and how the menu-bar icon looks. These were built and checked with stand-ins and dry runs only.
