# HAL voice project

This folder holds a voice interface to Claude Code: the user speaks, and replies are spoken aloud in a HAL 9000 voice. `README.md`, `LISTENER.md` and `notes/` describe how it works and what is planned (`notes/todo.md`, `notes/build-plan.md`).

## When this is the voice session (started with `bin/hal claude`)

**The first paragraph of every reply is what the user hears.** The voice speaks the opening paragraph of each reply and nothing else (`spoken_reply: "lead"` in `config.json`). So write that paragraph to be spoken:

- Answer the question in it. One to three short sentences, under about 450 characters.
- Plain prose only: no markdown, lists, code, file paths or URLs in that paragraph. Say names, numbers, days and times in a way that reads aloud well.
- It must stand alone: the listener may hear nothing else. If you need a decision, ask for it there.
- Never start a reply with a heading, a list or a table. If the opening paragraph is not usable, a smaller model summarises the reply instead, which is slower and less accurate.

Everything after the first paragraph is for the screen: detail, tables, sources. Tables, code blocks and source lists are never read aloud. Replies of about 350 characters or less are read in full.

To have one reply read in full (when asked to "walk me through" something), run `touch state/full_next` before finishing the turn. To keep one reply silent, write the current Unix time to `state/skip_next_reply`.

**While you work, the line before a tool call is spoken too** (`hal/progress.py`). So in a turn that uses tools:

- Before the first tool call, and whenever there's news, write one short plain sentence: what you are about to do, or what you just found ("Let me fetch tomorrow's forecast." "The tests pass; now the docs."). Keep it under about 160 characters, in whole sentences, with no markdown, lists or paths. A longer opening, or one that starts with a list, is not spoken.
- Don't repeat the acknowledgement the listener already heard ("Certainly, I'll check that"). Say something new.
- Put the answer itself in the final message, after the last tool call. Text written before a tool call is heard only as a short progress line.
- If nothing is said for 12 seconds, a stock phrase ("Still working on it.") fills the gap at the next tool call, at most twice a turn.

**Quick lookups, one call each.** Weather: `bin/weather [place] [day]` (place defaults to London; day is today, tomorrow, a weekday, weekend, week or a date). It prints the day's summary and the night, morning, afternoon and evening. Don't use web search for weather.

## Spoken requests

What the user says aloud is transcribed and typed into this session by the listener, and starts with a label: `[spoken 21:14] Claude, …`. A typed prompt has no label. Speech-to-text makes mistakes, so read a spoken request as possibly misheard. The listener marks what it doubts or changed:

- `word [?]`: the transcriber was unsure of that word.
- `word [? Name]`: it may have been Name, one of the user's names or terms. The word is as heard.
- `Name [heard as "words"]`: the listener changed what it heard to Name. Such corrections can be wrong; if Name makes no sense there, the heard words may be right.
- `[pause 12s]`: the user paused that long there before going on.
- `[spoken 21:14; interrupted the voice during "…"]`: the user spoke over your voice, which was saying that sentence. They heard roughly up to there, not the rest.

Take the likeliest reading in context. Ask about a doubtful word only when it changes what you would do; otherwise go ahead. Never read the label or the marks aloud, and don't quote them in the spoken opening paragraph (the voice skips them anyway).

## Working in this tree

- The hooks in `.claude/settings.json` run on every prompt, reply and tool call of the live voice session. Never leave the tree in a state where `bin/hal hook` or `bin/hal-listen hook` fails. Run `.venv/bin/python -m unittest discover -s tests` before and after changes.
- Running processes (the listener, the front desk) keep old code until restarted: `bin/hal-listen stop && bin/hal-listen start`, `bin/hal desk stop && bin/hal desk start`.
- The voice's dependencies are in `.venv`; the listener's are in `.venv-listen`. Do not install into `.venv` casually.
