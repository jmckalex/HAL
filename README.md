# HAL: a HAL 9000 voice for Claude Code

You speak to Claude Code with its built-in `/voice` dictation, and it answers aloud in HAL 9000's voice.

The full documentation, covering the voice, the front desk, the listener, interruption, autostart and every setting, is published at https://jmckalex.org/software/hal/.

This is the simple version. When a turn finishes, a `Stop` hook takes Claude's final message, strips out what shouldn't be read aloud (code, tables, links, markup), caps the length, and speaks it with a local [Piper](https://github.com/OHF-Voice/piper1-gpl) voice model ([campwill/HAL-9000-Piper-TTS](https://huggingface.co/campwill/HAL-9000-Piper-TTS)). When you send your next prompt, HAL stops talking. Nothing runs in the background between turns, unless you turn on the [front desk](#the-front-desk).

## Setup

macOS with Python 3.9 or newer. Dictation needs Claude Code signed in with a claude.ai account.

```sh
git clone https://github.com/jmckalex/HAL.git && cd HAL
./setup.sh                 # creates .venv, installs piper-tts, fetches the model if missing, runs tests
bin/hal say "Good afternoon. Everything is running smoothly."
```

Settings are read from `config.json`, which is yours and not in git: copy `config.example.json` to start one (every setting has a default, so the file is optional). The listener's list of names, `lexicon.txt`, works the same way, from `lexicon.example.txt`.

The voice model (63 MB) lives in `models/`. If it's missing, `scripts/fetch-model.sh` downloads it from HuggingFace and checks it against the checksums of the tested files.

## Use

1. Start `claude` in this directory and accept the workspace trust prompt. Claude Code doesn't run any hooks until you do. The hooks are in `.claude/settings.json`.
2. Turn on dictation: `/voice tap` (tap Space to start, tap again to send) or `/voice hold` (hold Space while you speak).
3. Speak. When Claude finishes, HAL reads the reply. If you start a new prompt, HAL stops.

About the voice settings: the `voice` object (`enabled`, `mode`, `autoSubmit`) is honoured in project and local settings as well as user settings. The settings reference gives its scope as "any file", and Claude Code's `/voice` command checks project and local values. However, `/voice` always writes to your user settings, and a project-level value overrides that, so `/voice off` would then refuse to turn dictation off in this project. For that reason this project doesn't set `voice`: run `/voice tap` once and it persists. `autoSubmit` only matters in hold mode. Tap mode already sends automatically once the transcript is at least three words.

### The voice session

Only one Claude Code session has the voice: the one you start with `bin/hal claude` (any arguments are passed on, e.g. `bin/hal claude --continue`). It sets `HAL_VOICE=1` in that session's environment, which the hooks check. Sessions started with plain `claude` stay silent and cannot interrupt HAL, even after a global install. Subagents never speak. Set `voice_sessions` to `"all"` in `config.json` to let every session speak.

### What HAL says

HAL does not read long replies out. When a reply's speakable text is longer than `summary_min_chars`, a small model (Claude Haiku, through a headless `claude -p` call with no tools or settings) writes a spoken summary of two or three sentences, and HAL says that; the full reply stays on screen. Shorter replies are read as they are. Writing the summary takes about 3 seconds, so speech starts roughly 4 seconds after a long reply appears. If the model call fails, HAL reads the opening of the reply instead.

On your next prompt, Claude is told what HAL actually said, so you can ask it to "elaborate on the second point" and it knows which point you heard. `hal last` prints that text. Set `spoken_reply` to `"full"` to go back to reading the cleaned reply itself.

As soon as you submit a prompt HAL acknowledges it ("One moment, please."), so a spoken request is never met with silence. A reply that arrives while the acknowledgement is playing waits for it to finish. The phrases are pre-rendered, so this is immediate. With the front desk on, the acknowledgement is written for each prompt instead, and purely social remarks are answered at once.

Saying or typing a short stop request such as "please stop reading" or "that's enough, thank you" silences HAL without an acknowledgement, and the reply to it is not spoken. Any other prompt also silences whatever is playing. Turns that Claude Code starts by itself (background task notifications, messages from other sessions) neither interrupt HAL nor get an acknowledgement.

### While Claude works

In a turn that uses tools, HAL doesn't stay silent until the reply. An async `PreToolUse` hook (`hal/progress.py`) runs at every tool call:

- **Claude's own line.** The sentence Claude writes just before a tool call ("Let me fetch tomorrow's forecast.") is spoken straight away. The hook reads it from the session transcript, which holds it before the tool runs. Only short prose is spoken: whole sentences up to `progress_max_chars`, not a list, heading or code. `CLAUDE.md` asks Claude to write such a line in the voice session.
- **A stock phrase after a silence.** If there's no new line and nothing has been said for `progress_cue_seconds` (12), one of `progress_phrases` ("Still working on it.") is said instead, at most twice a turn. It is checked only at tool calls, so a single long-running tool or long thinking stays silent.
- **Only turns whose reply will be spoken.** The prompt hook starts a turn. The reply starting, `hal stop`, a spoken "stop", mute or a newer prompt ends progress speech for it. A stop request, a remark the desk answered, a slash command, a subagent's tool calls and other sessions get none.
- **Never over anything else.** Progress lines play in the acknowledgement slot, one at a time. A newer line drops an older one that hasn't started, and the reply waits for a line that's playing (up to 6 s, then cuts it).

Set `speak_progress` to `false` and `progress_cue_seconds` to `0` to turn this off.

### The front desk

The front desk is off by default. Set `front_desk` to `true` in `config.json` to turn it on. It is a long-lived helper process that hears each prompt before Claude does. It keeps a small fast model (Claude Haiku, a `claude -p` process left running, so there is no start-up delay per prompt) and the voice loaded.

- **Social remarks** such as "Good morning, Claude", "Thank you, that was very helpful" or "Tell me a joke" are answered aloud by the desk itself ("Good morning, Dr. Alexander."). The prompt is blocked, so Claude takes no turn. Claude Code shows a notice instead of a reply. In `claude -p` it reads `UserPromptSubmit operation blocked by hook: Answered aloud by the front desk: "…"`, followed by `Original prompt: …`; the interactive display has not been checked. The next prompt that does reach Claude tells it about these exchanges, once, so the conversation stays continuous.
- **Everything else** goes through to Claude unchanged, and the desk speaks an acknowledgement in line with it while Claude works: "Certainly. I'll check tomorrow's forecast." It never answers, guesses or promises a result.
- **When in doubt, it hands on.** A short reply that may answer a question Claude has just asked ("OK", "sounds good", "yes, go ahead", "no", "the second one", "read that again") never reaches the desk, which can't see the question. It gets "Very well." and goes straight to Claude. The desk also says "Very well." to other remarks that only make sense in context.
- **Timing.** A prompt of up to `desk_wait_words` words (12) waits for the desk's decision, which is usually 1.3–2 s, so Claude starts that much later than without the desk. Speech starts about 1.8–2 s after you submit the prompt. Longer prompts are never social, so they don't wait: they go to Claude at once, and the acknowledgement follows when it is ready. If the desk has no decision within `desk_wait_seconds` (3 s), you get the canned acknowledgement, as without the desk.
- **Starting and stopping.** `hal claude` starts the desk. If it isn't running, the next prompt starts it and gets the canned acknowledgement meanwhile. It exits after `desk_idle_minutes` (120) without a prompt. Only one runs. It restarts its model if that dies, and after `desk_max_turns` prompts (6) it replaces the model with a fresh one, which bounds the model's conversation.
- **Without blocking.** With `desk_block` set to `false`, a social remark goes through to Claude as well. Claude is told the desk has already answered aloud, and asked to reply in a line at most. Its reply is not spoken.
- **Summaries.** While the desk runs, spoken summaries of long replies come from a second Haiku it keeps warm (`desk_summaries`). Speech for a long reply then starts about 2.4 s after it appears, rather than about 4 s. If that fails, the usual one-shot call is made.
- **Turn judgements.** For the always-on listener, a third warm Haiku (`desk_turns`) judges whether a spoken request is finished or the speaker paused mid-thought (`{"op": "turn"}` on the socket; `hal desk turn TEXT` to try it). It speaks nothing, and the log records only the number of words. See LISTENER.md, "Turns".
- **Name.** `user_name` is how the desk addresses you, in a greeting or a farewell. It answers to "Claude" and never calls itself HAL.

**Costs.** Each prompt that reaches the desk is one Haiku call, and so is each summary, as before. Two `claude` processes stay running while the desk is up, using about 230 MB each.

`scripts/desk-check.py` runs a set of remarks through the real hook and the real model, silently, with a temporary state directory. It reports what the desk did with each one, and timings. Use it after changing the desk's instructions (`desk_prompt()` in `hal/desk.py`). It makes about 23 Haiku calls.

### Commands

`bin/hal` controls the voice. To use it from anywhere, link it onto your PATH, for example `ln -s "$PWD/bin/hal" ~/bin/hal`.

| Command | Effect |
| --- | --- |
| `hal off` | Mute and stop speaking. The hooks do nothing until `hal on`. |
| `hal on` | Unmute. |
| `hal stop` | Stop the current speech and stay unmuted. |
| `hal last` | Print what HAL last said in place of a full reply. |
| `hal status` | Show whether HAL is muted or speaking, whether the model is present, and the current settings. |
| `hal say TEXT` | Speak TEXT, or stdin, cleaned and capped like a reply. This works while muted. `--wait` blocks until speech ends (Ctrl-C stops it), `--wav FILE` renders to a file instead, and `--raw` skips cleaning and the cap (pronunciation fixes still apply). |
| `hal clean` | Print what would be spoken for the Markdown on stdin. |
| `hal desk start` / `stop` / `status` | Start the front desk in the background, stop it, or show its state. `hal desk run` runs it in the foreground. |
| `hal desk ask TEXT` | Show what the front desk would do with TEXT, without speaking. |

Only one voice speaks at a time across all sessions: a new reply cuts off the previous one.

`bin/weather [place] [day]` prints a forecast in one call (Open-Meteo, no key), so a spoken question about the weather needs one quick step instead of a web search. `CLAUDE.md` tells the voice session to use it. See `bin/weather --help`.

### Output style

`.claude/output-styles/hal.md` defines an output style called **HAL**. It asks Claude to open every reply with a short, plain-prose summary that works when heard (calm, measured, a light trace of HAL's manner), and to put the code and detail after it. It keeps Claude Code's normal coding instructions.

It is not turned on. To try it, run `/output-style HAL`. To make it the default for this project, add `"outputStyle": "HAL"` to `.claude/settings.local.json`. Claude Code reads style files at startup, so restart it if you edit the file mid-session.

### Settings

`config.json`:

| Key | Default | Meaning |
| --- | --- | --- |
| `max_chars` | `600` | Spoken text is cut at a sentence boundary after this many characters, then HAL says the overflow message. `0` removes the cap. HAL speaks slowly, about 14–17 characters a second, so 600 characters is roughly 35–45 seconds. |
| `overflow_message` | `"The rest is on your screen."` | What HAL says after cutting a reply short. |
| `length_scale` | `null` | Speech rate. Values above 1 are slower and below 1 faster. `null` uses the model's own rate. |
| `speak_in_print_mode` | `false` | Also speak replies from `claude -p` and Agent SDK runs. |
| `spoken_reply` | `"summary"` | `"summary"`: speak a short model-written summary of long replies. `"full"`: speak the cleaned reply itself, up to `max_chars`. |
| `summary_min_chars` | `350` | Replies whose speakable text is no longer than this are read in full, without the model. |
| `summary_model` | `"haiku"` | Model alias passed to `claude -p --model` for the summary. |
| `summary_timeout` | `20` | Seconds to wait for the summary before falling back to the reply's opening. |
| `acknowledge` | `true` | Speak a brief acknowledgement when a prompt is submitted. |
| `ack_phrases` | four phrases | The acknowledgements, chosen at random without repeating the previous one. |
| `voice_sessions` | `"master"` | `"master"`: only a Claude Code session started with `hal claude` speaks; every other session's hooks do nothing. `"all"`: every session speaks. |
| `front_desk` | `false` | Turn on the [front desk](#the-front-desk). With `false`, nothing below has any effect. |
| `desk_block` | `true` | Block a social remark the desk has answered, so Claude takes no turn. `false`: pass it on, and don't speak Claude's reply to it. |
| `desk_wait_words` | `12` | Prompts of at most this many words wait for the desk's decision. Longer ones go straight through. `0`: never wait, so nothing is answered by the desk. |
| `desk_wait_seconds` | `3.0` | How long such a prompt waits before falling back to the canned acknowledgement (0.5 to 3.5; the hook's own timeout is 5 s). |
| `desk_summaries` | `true` | Write spoken summaries with a warm model in the desk. |
| `desk_turns` | `true` | Keep a third warm model that judges, for the always-on listener, whether a spoken request is finished (LISTENER.md, "Turns"). |
| `desk_max_turns` | `6` | Replace the desk's model with a fresh one after this many prompts. `1` gives no history at all, but a quick exchange is then about 0.5 s slower per prompt. |
| `desk_idle_minutes` | `120` | The desk exits after this long without a prompt. |
| `desk_model` | `"haiku"` | Model alias for the desk. Summaries use `summary_model`. |
| `user_name` | `""` | How the desk addresses you, e.g. `"Dr. Alexander"`. Empty: no name. |
| `speak_progress` | `true` | Speak the line Claude writes before a tool call ([While Claude works](#while-claude-works)). |
| `progress_max_chars` | `160` | Longest progress line, in whole sentences. A first sentence longer than this isn't spoken. |
| `progress_cue_seconds` | `12` | At a tool call after this long with nothing said, speak a stock phrase. `0`: never. |
| `progress_phrases` | two phrases | The stock phrases, used in order, one per silence, at most two a turn. |

#### Pronunciation

Piper's espeak-ng front end spells out an all-caps word inside a mixed-case sentence. That's right for API, JSON, URL or CLI, but wrong for a name: "This is HAL speaking" comes out as "aitch-ay-el speaking". Just before synthesis, `PRONUNCIATIONS` in `hal/textclean.py` respells such words. It matches whole words only, is case-sensitive, and keeps a plural "s". Only the spoken text changes; what's on screen stays the same.

The current entries are:

| Written | Spoken |
| --- | --- |
| HAL | Hal |
| WAV | wave |
| RAM | ram |
| DOM | dom |
| ENV | env |
| ONNX | onyx |
| STDIN | standard in |
| STDOUT | standard out |
| STDERR | standard error |

To add a word, add an entry to `PRONUNCIATIONS`. Run `hal clean` to see what will be said.

## Make it global

To have HAL speak in every project, register the same hooks in `~/.claude/settings.json` using absolute paths:

```sh
./install.sh --dry-run     # show what would change
./install.sh               # add the hooks (backs up settings.json first)
./install.sh --uninstall   # remove them
```

The script merges rather than overwrites. It touches only HAL's own two hook entries, leaves your other settings and hooks alone, and is safe to run repeatedly. Before each change it saves the old file as `settings.json.bak-hal-<timestamp>`. It re-indents the file with two spaces. If `settings.json` is a symlink, the script updates the file the link points to. Restart Claude Code afterwards.

With the global hooks installed, both the global and the project hooks fire inside this directory. That's harmless: the second one replaces the first within milliseconds, before any audio plays. You can also delete the `hooks` block from `.claude/settings.json`.

## How it works

```
Stop hook ──► bin/hal hook ──► hal/hook.py: clean + cap (hal/textclean.py) ──► spawn detached worker ──► exit (~50 ms)
                                                                                  │
                                                         hal/speaker.py worker: Piper → WAV → afplay
UserPromptSubmit hook ──► bin/hal hook ──► kill the worker's process group (afplay included); start a turn
PreToolUse hook (async) ──► bin/hal hook ──► hal/progress.py: Claude's latest line from the transcript ──► worker (ack slot)

With the front desk on:
UserPromptSubmit hook ──► bin/hal hook ──► stop speech ──► Unix socket ──► hal desk (long-lived: Haiku + Piper loaded)
        ▲                                                                   │ decides: social reply or acknowledgement
        └──── block the prompt, or let it through ◄─────────────────────────┤
                                                                            └► render WAV ──► worker plays it (same slots)
```

- **The `Stop` hook is `async`**, so Claude Code never waits on speech. The `UserPromptSubmit` hook is synchronous with a 5-second timeout: it returns in about 50 ms, and its only output is the one-line note telling Claude what was spoken. Both always exit 0. Errors go to `state/hal.log`.
- **The worker runs in its own session and process group**, so the audio outlives the hook process, even if Claude Code kills the hook's process group.
- **Only one utterance plays at a time.** A ticket file and a PID file under an `flock` in `state/` coordinate this. A worker that was stopped or superseded while still synthesising never starts playing.
- **The modules are separate so a future daemon can reuse them.** `hal/textclean.py` is pure and standard-library only, and `hal/speaker.py` has `synthesize()`, `speak()` and `stop()`.
- **Deep install paths are handled.** espeak-ng can't use a data path of 160 characters or more. If the project sits deep enough that Piper's bundled `espeak-ng-data` path is that long, `synthesize()` reaches it through a short symlink in the temp directory.

- **The front desk speaks through the same slots.** A social reply takes the reply slot, and an acknowledgement takes the acknowledgement slot. The desk renders the WAV with the voice it keeps loaded, and a short-lived worker plays it, so `hal stop`, mute and a newer prompt silence it as they silence anything else. Each prompt's speech carries the token the hook's stop left in `state/current`. If anything has replaced it before the audio is ready (a newer prompt, `hal stop`, mute, a reply that has already started), the desk drops it.
- **The desk's protocol** is one line of JSON each way over `state/desk.sock`; see `hal/desk.py`. A short prompt waits until the model's first word shows which kind of answer it is. An acknowledgement releases the prompt at once; a social reply waits for the whole reply. The hook confirms the decision before the desk speaks, so a decision that arrives after the hook has given up is never spoken.

All runtime state is in `state/`: the mute flag, the lock, the current ticket and PID, temporary WAVs, and `hal.log`. With the front desk on, `desk.sock`, `desk.pid` and `desk.lock` belong to it, and `desk_exchanges.json` holds social exchanges that Claude hasn't been told about yet.

## Tests

```sh
.venv/bin/python -m unittest discover -s tests -v
```

`tests/test_textclean.py` holds unit tests for the cleaner. `tests/test_desk.py` tests the front desk the same way, with `tests/fake_warm_model.py` standing in for Haiku. `tests/test_hook.py` holds end-to-end tests that pipe Stop and UserPromptSubmit JSON through the exact command in `.claude/settings.json`. These tests synthesise real audio but "play" it with a silent stand-in player (`tests/fake_player.py`) and use a temporary state directory. They cover hook timing, audio output, cut-off, stop, the stop-during-synthesis race, mute, malformed input, subagent and print-mode events, and a missing venv.

## Known limits

- **Latency.** A short reply starts about 0.7–0.8 s after Claude finishes. A long reply starts about 4 s after, because the summary is written first (about 3 s) and then synthesised. That drops to about 2.4 s with the front desk running. The model is loaded fresh for every reply.
- **The front desk's choices are a model's judgement.** In the checks so far it never answered a remark that needed Claude, and short answers that might reply to a question never reach it. Its wording varies, though: it sometimes uses your name outside a greeting, or makes a joke where a plain acknowledgement would do. A blocked prompt's text still goes into Claude Code's transcript and prompt history.
- **The summary needs your Claude sign-in.** It is written by `claude -p`, which uses the same login as Claude Code. If that login has expired, the call fails and HAL reads the opening of the reply instead.
- **Between tool calls, only short lines.** The reply is `last_assistant_message`. Text Claude writes before a tool call is spoken only as a short progress line (whole sentences up to `progress_max_chars`), and it is read from the transcript file, which Claude Code's docs say may lag behind. A line that reaches the file late is spoken at the next tool call, or not at all. Subagent output and permission prompts are never spoken.
- **No speech when a turn doesn't end normally.** If you interrupt with Esc, no `Stop` event fires. API errors fire `StopFailure` instead, which HAL ignores.
- **HAL keeps talking while you dictate.** Pressing Esc or starting to dictate doesn't stop HAL; only submitting a prompt does. Your microphone can pick up HAL's voice, so use headphones or run `hal stop` first.
- **Detecting who started a turn is a heuristic.** `UserPromptSubmit` also fires for task notifications, messages from other sessions and scheduled prompts. HAL treats a prompt that starts with `<` or `[`, or with "Another Claude session", as not yours and leaves it alone. The log records the first characters of anything it skipped, so a wrong guess is easy to spot.
- **Print-mode detection is inferred.** `claude -p` and SDK sessions are kept silent by checking `CLAUDE_CODE_ENTRYPOINT`, which starts with `sdk-` there and is `cli` in the interactive CLI. That behaviour was read from the Claude Code 2.1.325 binary, not from the docs, so it could change.
- **Source lists are not read.** A "Sources" or "References" label followed by links or list items, or a "Source: [link]" line, is left on screen and skipped in speech.
- **The cleaner is heuristic and English-only.** Long inline code (more than six words) is dropped, so "Run `<long command>` to check" becomes "Run to check". File paths are reduced to the file name. Unusual Markdown may come out oddly. Use `hal clean` to see exactly what would be said.
- **macOS only as written**, because playback uses `afplay`. Setting `HAL_PLAYER` to another command that takes a WAV path would also work.
- **The model isn't in git.** `.gitignore` excludes `models/`, and `scripts/fetch-model.sh` restores it. Check the model's licence on HuggingFace before redistributing it.

## Licence

MIT; see `LICENSE`. The voice model, Piper and the speech models the listener downloads have their own licences.
