# Always listening: the HAL front end

With the listener running, you don't press anything to talk to Claude. The laptop microphone is always on. When you say something addressed to Claude ("Good morning, Claude", "Claude, what's the weather tomorrow?", "By the way, Claude, open my thesis"), the whole request is typed into the running Claude Code session and Enter is pressed, exactly as if you had typed it. Everything else said in the room is ignored. The existing hooks then acknowledge and answer aloud as before.

You can stop to think in the middle of a request, for as long as a minute. The listener decides that you have finished by what you said, not by how long you were silent: "Claude, could you look at the draft of my paper and … [thirty seconds] … check the second section?" is typed as one line once you have finished it (see [Turns](#turns-when-have-you-finished)).

Everything runs locally: Silero VAD finds the utterances, Parakeet (on MLX) transcribes each one, and a text rule decides whether it was addressed to Claude. Audio stays in memory. The one exception is the judgement of whether a request is finished: when the rules can't tell, the text of that request (and nothing else) goes to the front desk's Claude Haiku, which sees only what is about to be typed into Claude Code anyway.

## Setup

```sh
scripts/setup-listen.sh      # .venv-listen, the two models, the listener's tests
bin/hal-listen mic-check     # 3 s of levels from the microphone (nothing kept or transcribed)
duplex/build.sh              # only for duplex "aec": the audio helper (swiftc, a few seconds)
```

- **`.venv-listen`** is separate from the voice's `.venv`, so installing or upgrading the listener can never break speech. It holds `parakeet-mlx`, `mlx`, `sounddevice`, `onnxruntime` and `numpy`, pinned in `requirements-listen.txt`.
- **`models/silero_vad.onnx`** (2.3 MB, MIT licence) comes from the `silero-vad` 6.2.0 wheel on PyPI and is checked against its SHA-256. The wheel itself isn't installed because it needs torch.
- **`models/parakeet-tdt-0.6b-v2/`** (2.5 GB, NVIDIA's CC-BY-4.0 model in the `mlx-community` conversion) is cloned from `~/.cache/huggingface` if it is already there. The clone is an APFS copy-on-write clone, so it is instant and takes no extra disk space. Otherwise it is downloaded into `models/`. Downloads never go to `~/.cache`: `HF_HOME` points to `models/hf/`.
- The terminal app needs microphone permission (System Settings, Privacy & Security, Microphone). Without it macOS delivers silence, and `mic-check` says so.

## Use

Start Claude Code with `hal claude` as usual (after the integration steps at the end of this file). Claude Code then runs under a small pseudo-terminal proxy, and the listener starts in the background and serves that session. Nothing looks different at the keyboard. Then just talk:

| You say | What happens |
| --- | --- |
| "Claude, what's the weather tomorrow?" | Typed into the session and sent. HAL acknowledges and answers as usual. |
| "Good morning, Claude." / "By the way, Claude, open my thesis." | Typed whole, including the words before the name. |
| "Claude, could you look at the draft and …" (pause to think) "… the second chapter?" | Held while it is unfinished, then typed as one line. What follows the pause needs no name. |
| "… That's all." / "… Go ahead." | Sends a request being held at once, without those words. |
| "… Never mind." / "… Cancel that." | Discards a request being held. "Scratch that. Open my thesis." starts it afresh. |
| "Claude?" or "Hey Claude." | HAL answers "Yes?" (`name_only_reply`, which can include your name: "Yes, Dr. Alexander?"), and your next utterance needs no name. |
| "Claude, um …" (the name, then a sign of more) | No answer: the listener waits for the request, which needs no name. |
| Anything, while Claude is still working on your spoken request | Typed as part of the conversation, without the name (`during_turn`). Claude Code queues it for the model. |
| Anything, over HAL's voice (only with `duplex` `"aec"`) | The voice stops within about half a second. "Stop", "that's enough" or "thank you, Claude" only silence it; anything else is also taken as a request, without the name. "Go on" brings it back. See [Interrupting the voice](#interrupting-the-voice-duplex-aec). |
| "Claude, stop listening." | HAL says it is no longer listening, and the microphone closes. Resume from the keyboard with `hal-listen resume` (in Claude Code: `! bin/hal-listen resume`). |
| Anything not addressed to Claude | Ignored. The log records only that something was ignored and how long it was. |

If Claude Code is showing a permission prompt or another dialog when you speak, nothing is typed, because the characters would answer the dialog. HAL says "I need you at the keyboard." Your request is held and sent once the dialog closes, or dropped after two minutes.

### Commands

| Command | Effect |
| --- | --- |
| `hal-listen run` | Listen in the foreground and show what is heard and decided. On a terminal this includes the words of ignored utterances; they are shown, never logged. `--dry-run` decides without typing (no session needed). `--silent` prints what HAL would say instead of saying it. `--hide-ignored` hides the ignored words. `--device` picks the input. `--duplex aec` or `half` overrides the setting. |
| `hal-listen start` / `stop` | Start or stop the background listener. Only one listener runs at a time. It also stops by itself when the session it serves exits. |
| `hal-listen pause` / `resume` | Close the microphone, or open it again. |
| `hal-listen status` | The listener, the session (turn, open dialogs, whether typing is allowed now), the models and the key settings. |
| `hal-listen devices` | List the audio inputs. |
| `hal-listen mic-check` | Open the microphone for a few seconds and print levels and seconds of speech only. |
| `hal-listen type TEXT` | Type TEXT into the session with the same safety checks as speech. Useful for checking the wiring. |
| `hal-listen check` | Say whether text may be typed into the session right now, and if not, why. |

The log is `state/listen.log`.

## How it works

```
microphone ─► 32 ms frames ─► voice gate ─► Silero VAD ─► segmenter ─► Parakeet ─► addressing rule
   (sounddevice, 16 kHz)     (drop while        (speech          (utterances)   (≈70 ms)        │
                              HAL speaks)        probability)                                    ▼
                                                                    injector ─► Unix socket ─► pty proxy ─► claude
                                                                    (holds while a dialog is open)   ▲
Claude Code hooks ─► bin/hal-listen hook ─► state/listen/session-<proxy>.json ───────────────────────┘
```

### Hearing utterances

`segmenter.py` follows Silero's own VADIterator, with settings tuned for spoken requests:

- **Start.** An utterance starts at the first frame with speech probability ≥ 0.5. The 0.4 s of audio before it (`pre_roll`) is included, so a soft onset is never clipped.
- **End.** It ends after 0.8 s with no frame ≥ 0.5 (`end_silence`). The count starts at the first frame below 0.35, so a frame in between neither starts nor stops it. 0.2 s of that silence is kept. 0.8 s keeps a short pause after the name ("Claude, … what's the weather") inside one utterance. With turns (the default), an utterance is only a fragment: whether the request is finished is decided afterwards, by what was said.
- **Noise.** A burst with less than 0.25 s of speech frames is dropped.
- **Long speech.** At 30 s an utterance is cut, and the parts are transcribed separately and joined before the decision. A long dictation that names Claude anywhere is typed whole.

### Deciding

`addressing.py` matches whole words, ignoring case. Speech-to-text doesn't always write "Claude", so it accepts two kinds of variant:

- **`names`** count anywhere: `claude klaud klaude clawd klawd clode klode cloat cload gload`. None of them is an English word.
- **`vocative_names`** count only where a name is being called: `claud cloud clod clawed claw clause clothes cloth cloak clog`. The word must be followed by punctuation or the end of the utterance. It must also be preceded by a comma, or by nothing but lead words ("hey", "good morning", "thanks", …) since the start of the sentence.
  - These match: "Cloud, what's the weather?", "Thanks, cloud.", "By the way, cloud, …", "Hey cloud."
  - These don't: "stored in the cloud", "cloud storage", "the cat clawed at the door".
  - The matched word is written "Claude" in the text that is typed.

The variant list comes from Parakeet's mistakes on synthetic voices: "Cloat", "Cload", "Gload", "Clothes", "Cloth", "Cloak", "a cloud". Parakeet never wrote an ordinary "cloud" sentence as "Claude". "Claud" is in the second list because one voice's "clod" came out as "claud".

A third kind comes from the lexicon's aliases for the name (`god -> Claude @calling` in `lexicon.txt`; see [the lexicon](#names-the-transcriber-gets-wrong-the-lexicon)): `calling_names`, "God", "Todd" and "Gaud", heard for a real voice. They are stricter than `vocative_names`: also not after "oh", "my", "thank", "good" and the like, not followed by "!", and not alone. Unlike the variants above, the rewrite is shown in what is typed: `Claude [heard as "God"]`.

`decider.py` adds the following:

- **The name alone.** "Claude.", "Hey Claude?" or "Okay Claude" makes HAL answer "Yes?" (`name_only_reply`).
  - The listener first waits 0.25 s (`name_only_grace`). If you are already speaking again, it doesn't answer, and it takes that speech as the request.
  - Speech that starts within 8 s (`armed_seconds`) after the answer needs no name. That window counts from when the answer finishes.
- **Follow-up window.** Set `follow_up_seconds` to accept, without the name, speech that starts within that many seconds after HAL finishes speaking a reply. The default is 0, which is off. Acknowledgements and the listener's own phrases don't open the window.
- **While Claude works** (`during_turn`, on by default). After the listener has typed a request, and until that turn of the session ends (the `Stop` hook), speech needs no name: it is taken as part of the conversation and typed, and Claude Code delivers it to the model mid-turn. "Until the turn ends" comes from the session-state file the hooks keep: the turn is `busy` and has not stopped since the listener typed. A turn you started at the keyboard doesn't count, nor does one that never reported its end after 30 minutes. The voice gate still applies (nothing is heard while HAL speaks), and once the turn has ended the follow-up window takes over.
- **Continuing a turn.** While a request is being held unfinished (see Turns), what you say next joins it without the name.
- **"Claude, stop listening"** (also "pause listening", with or without "please" or "for now") pauses the listener. It isn't typed.

### Turns: when have you finished?

With `turn_taking` `"semantic"` (the default), a pause yields a fragment, not a request (`turns.py`). An addressed fragment starts a *turn*, which is held; everything said before the turn is sent joins it without the name. After each fragment the whole turn so far is judged:

| Verdict | What happens |
| --- | --- |
| finished | Sent once 1.2 s of silence has followed it (`turn_end_silence`). Speech that resumes before then still joins it. |
| unfinished | Held. Sent as it stands after 60 s of silence (`turn_max_wait`). |
| unclear | The front desk's warm Claude Haiku judges it (`turn_judge`, normally within a second). If the desk is off, not running or slower than 2.5 s (`turn_judge_timeout`), the turn is sent after 3 s of silence (`turn_unclear_wait`). A desk that isn't running is started then for the next turn, and whenever listening resumes. |

The rules come first. They are cheap, and they decide only what they can decide reliably:

- **Unfinished:** an ellipsis, comma, colon or dash at the end; a last word that can't end a sentence (and, or, but, because, if, than, the, a, my, your, …); a preposition at the end without a full stop ("… lengthen the pause for"; Parakeet punctuates from the voice, so a finished "the book I was looking for." gets its stop); endings such as "the thing is", "such as", "and then", "but for the moment"; only filler ("So.").
- **Finished** (only with a final `.`, `?` or `!`): a question that doesn't break off and isn't a preamble ("Can I ask you something?" goes to the judge); a short imperative or polite request ("Open my thesis.", "Please read me the summary.", "Can you open the second chapter."); a greeting, thanks or brief answer ("Thanks, Claude.", "Yes.", "It's very helpful.").
- Everything else is unclear and goes to the judge: statements ("I think the conclusion is weak."), long requests, announcements ("I have a question about the paper.").

**The judge** is a request on the front desk's socket, `{"op": "turn", "text", "timeout"}`, answered `{"kind": "complete" | "incomplete"}` or `{"kind": "unavailable", "why"}` (`hal/desk.py`, client in `judge.py`). The desk keeps a third warm Haiku for it, with its own instructions (`desk.turn_prompt()`), apart from its acknowledgement and summary models, and replaces it after 12 judgements. Its answer is one word, and the desk replies as soon as the first three letters show which. The listener asks in a background thread and polls, so listening never waits for it. `desk_turns: false` in the top level of `config.json` turns the desk's side off.

**Spoken controls,** in any fragment after the first:

- A send phrase (`turn_send_phrases`: "that's all", "that's it", "go ahead", "send it", "over to you", …), on its own or ending the fragment after punctuation, sends the turn at once, without the phrase.
- A cancel phrase (`turn_cancel_phrases`: "never mind", "cancel that", "forget it", "scratch that", …), on its own or ending the fragment, discards the turn. At the start of a fragment, followed by more ("Scratch that. Open my thesis instead."), it discards what came before, and the rest starts the turn afresh.
- In the first fragment these are ordinary words: "Yes, go ahead." answers a question and is typed as it is (a send phrase at its end only marks it finished), and "Claude, never mind." is typed as before.

**The name, then a pause.** "Claude." or "Claude?" alone is answered as before, after `name_only_grace`. "Claude," "Claude…" or "Claude, um" (a mark that more is coming) is not answered: it opens an empty turn, so the request that follows needs no name. If nothing follows within `turn_max_wait`, it is dropped. This depends on Parakeet writing the comma, ellipsis or filler; for synthetic voices it wrote "Claude." every time, so don't count on it. A longer `name_only_grace` gives a pause after the name more room.

**Transcribed afresh.** Once a turn has two or more fragments, their audio (kept in memory, at most 60 s) is transcribed again as one piece, with 0.3 s of silence between the pieces, for better words and punctuation across the pauses (`turn_retranscribe`). That takes about 0.4 s for 60 s of audio. If the result loses most of the words, the pieces' own transcripts are joined instead.

**What you hear while it waits:** nothing, by default (`turn_cue` `"none"`). `"sound"` plays a soft system sound (`turn_cue_sound`, at a quarter volume) once per turn, after `turn_cue_after` (10 s) of silence. `"words"` says `turn_cue_words` ("Go on.") instead, but anything HAL says closes the microphone while it plays, so speech that starts during it is lost. Cancelling says nothing unless `turn_cancel_reply` is set, for the same reason.

**Other behaviours carry on:** "Claude, stop listening" pauses and discards a turn being held, as does `hal-listen pause`; the voice gate drops speech while HAL speaks, and the turn keeps waiting; a long utterance cut at 30 s is joined first and counts as one fragment; a sent turn goes through the injector like any request (held while a dialog or draft is open, typed in order).

`turn_taking: "silence"` restores the behaviour before turns: every utterance is a request, and only `end_silence` decides when you have finished.

### What the model is told

Everything the listener types starts with a label, and may carry marks, so that Claude knows the request was spoken, when, and which words to doubt. `hal/spoken.py` defines the format; `annotate.py` builds it when a turn is sent. Three real examples (synthetic voices, real Parakeet):

```
[spoken 23:55] Claude, please ask Clew-boss [heard as "ClueBoss"] to add a line break in the text plus manual.
[spoken 23:55] Claude, I'd like you to compare the two drafts of my paper, and [pause 6s] then send the better one to Isabel [? Isobel] and Robin.
[spoken 23:55] Claude, check the Jmapdown [?] repository for Haslinger [? Haslanger] and Bicchieri [heard as "Bakiri"].
```

| Mark | Meaning |
| --- | --- |
| `[spoken 21:14]` | Spoken, starting at that time (local, 24-hour). Always first. |
| `[spoken 21:14; interrupted the voice during "…"]` | … and the speech cut off the voice, which was saying that sentence (at most 90 characters). Only when it did (duplex `"aec"`), and not when the voice had already said everything. |
| `word [?]` | The transcriber was unsure of the word. |
| `word [? Name]` | Unsure, or it sounds just like one of your names: perhaps Name. The word is as heard. |
| `Name [heard as "words"]` | The listener changed "words" to Name (the lexicon, below; also the name "Claude" heard as "God"). |
| `[pause 12s]` | You paused that long there, between two fragments of a turn. |

- **Nothing heard can pass for a mark.** Square brackets in a transcript become round ones before anything else sees it (`spoken.sanitize()`; Parakeet doesn't write them anyway). In a spoken request, every bracketed part was added by the listener.
- **The words alone.** `spoken.plain()` removes the label and marks from a prompt that starts with the label, and leaves any other prompt alone. The hook's checks (is it the user's prompt, a stop request, a social remark, a brief answer), the prompt it saves for spoken summaries, and the front desk (prompts, turn judgements, `hal desk ask`) all use the words alone, so a spoken request is treated exactly as the same words typed. The words are the corrected ones.
- **Never read aloud.** `textclean.to_speech()` removes these marks wherever they appear, so the voice skips them even if a reply quotes one.
- **Logged** like any addressed request: `listen.log` records the typed text, with a `marked:` line listing corrections, flags, doubtful words and pauses. `listener.json` counts them (`corrected`, `flagged`, `doubtful`).
- `CLAUDE.md` tells Claude how to read them.

**Doubtful words.** parakeet-mlx 0.7.0 gives each token of its greedy decoding a `confidence`: one minus the entropy of the model's distribution over the vocabulary and blank at that step, divided by its maximum (`parakeet.py`, `decode_greedy`). It gives no alternative words: its beam search (`Beam`) returns only the best hypothesis. `asr.transcribe_words()` groups the tokens into words, each with the lowest confidence of its letter-bearing tokens. A word below `doubt_threshold` (0.82) is marked `[?]`: at most the two least certain in a sentence, never a short function word ("the", "could"), the name, or one of your names. A turn transcribed afresh is marked from that transcription. Calibrated on synthetic speech (`say -o`, six natural voices, fifteen everyday sentences, and the robotic voices for hard speech): clearly heard words scored 0.9–1.0, misheard names 0.56–0.87; at 0.82, one of 90 clear sentences got a mark ("pi [?]"), and the hard sentences got one or two each, mostly on the wrong words.

### Names the transcriber gets wrong: the lexicon

Parakeet doesn't know your names: "Clew-boss" arrived as "Clue Boss", "TeX+" as "Tech Plus", "Kolodzin" as "Culdran". The lexicon (`lexicon.py`) corrects them, in addressed speech only, when a turn is sent; the turn judge sees the words as heard.

- **Sources** (read-only, local, kept in memory, never sent anywhere): `lexicon.txt` in the project, edited by hand, and the names of the folders directly inside each of `lexicon_dirs`, such as a folder of projects and a paper library's author folders (three such folders gave about 1,430 names). The file is re-read within 5 s of a change; the folders every ten minutes.
- **`lexicon.txt`**: one name per line, optionally with how it is said (`TeX+ | tech plus, tex plus`), and aliases, mistakes seen in your speech: `clue boss -> Clew-boss` (always applied), `robbing -> Robin ?` (only flagged: `robbing [? Robin]`), and `god -> Claude @calling` (the name, only where it is being called; see below).
- **Sounds alike.** Otherwise a word, or two or three, is compared with every name by a rough English pronunciation key (`phonetic()`: consonant classes, vowels collapsed, letters of acronyms spelled out) and a weighted edit distance. It is changed only when it is not an ordinary word in context:
  - an unfamiliar word (not in `/usr/share/dict/words`, its lower-case words and their inflections, nor its known names such as "Monday" or "Isabel"), which the transcriber was also less than 90% sure of: "Kolodzen" (0.76) becomes Kolodzin; "Bob Marley", heard with confidence, is only flagged `Marley [? Marlow]`;
  - or two or three words mangled into one name, written as a name: "Clue Boss", "Edi Amin", "MP ticks WASM".
  A close but less certain match, a known name that sounds just like one of yours ("Isabel [? Isobel]"), and a doubtful long ordinary word that sounds just like one of yours are only flagged. Folder names (1,400 of them) are considered only for words the transcriber was unsure of. Ordinary words are left alone: "cloud storage", "hacking the server", "go to the shop", "a line break".
- **Run together.** Two or three words that spell a name when joined: "ghost script" becomes ghostscript.
- **The name.** Aliases for "Claude" go to the addressing rule (`calling_names`): "God", "Todd" and "Gaud" count as the name only in the calling position and not as an exclamation: "God, what's the time?", "Good morning, God." and "Thanks, God." do; "Oh God.", "Thank God.", "My God!", "God." alone and "Todd and I" don't. "God, I'm tired." does, and is typed `Claude [heard as "God"], I'm tired.`, so Claude can tell.
- **Measured** (synthetic voices; `tests/test_listen_spoken.py` has the cases): every mistake listed in `lexicon.txt` is corrected (by its alias); of 30 sentences with names from the lexicon spoken by three voices, 21 misheard names were corrected, all rightly, 8 flagged with the right name and 2 marked doubtful; of 46 ordinary sentences with look-alike words, phonetic matching changed none and flagged 4 (Marley, a common first name, and Haslinger twice); two aliases made from ordinary words changed 4 ordinary uses (an alias from "imaging" also fires on "the imaging lab"). Mark an alias `?` if it misfires.

**Pauses.** Where two fragments of a turn were separated by at least `pause_mark_seconds` (3 s) of silence, `[pause Ns]` is put between them (after the last word of the first, matched through the words the transcriber produced, also when the turn was transcribed afresh).

**Interruptions.** When the utterance that becomes (part of) a request cut the voice off (duplex `"aec"`: speech over it, or a named request said over it), the label says which sentence the voice was in, from `Interrupted.reached` (the same record as `interrupted.json`).

### Not hearing itself

With `duplex` `"half"` (the default), `gate.py` polls `state/speaker.pid` and `state/ack.pid`, the PID files `hal/speaker.py` keeps for the worker that is speaking.

- A worker's file appears before it synthesises and is removed after playback. While either worker is alive, and for 0.6 s afterwards (`echo_tail`), captured audio is dropped unheard.
- An utterance that the voice interrupts is discarded whole.
- This is half-duplex: you can't interrupt HAL by voice while it speaks.
- After a typed request, the acknowledgement and then the reply keep the gate closed for about 8 s after you stop speaking (less for short replies).

With `duplex` `"aec"` the microphone's audio is echo-cancelled instead, and the gate applies only when that isn't possible (next section).

### Interrupting the voice (duplex "aec")

Off by default. With `"listener": {"duplex": "aec"}` in `config.json` the listener keeps listening while HAL speaks, through the laptop speakers or anything else, and speaking over the voice interrupts it.

**Echo cancellation.** `duplex/hal-duplex` (Swift, `duplex/HALDuplex.swift`, built by `duplex/build.sh` with `swiftc` and nothing else) captures the default input with Apple's voice processing (the VoiceProcessingIO unit, through AVAudioEngine) and passes 16 kHz frames to the listener over a pipe (`duplex.py`). On macOS the canceller's reference is everything the output device plays, from any app. Apple's header for `setVoiceProcessingEnabled` says it removes "any of the audio that is played from the device", and the measurements below confirm it for `afplay` and for the helper's own playback. HAL's voice is still played through the helper: `hal.config.player_command()` uses `hal-duplex play` while the helper's socket `state/listen/duplex.sock` exists, and otherwise `afplay` as before. Through the helper the voice isn't ducked as "other audio", can be paused and resumed at once, and its position is known. If the helper doesn't answer, `play` runs `afplay` itself, so the voice is never lost.

**Devices.** The helper always uses the system's current default input and output. It never selects or changes a device; `device` is ignored in this mode. When either default changes (AirPods connect, headphones are plugged in, back to the built-in speakers), the helper rebuilds on the new devices and carries on, and a reply that was playing continues from where it was. With headphones as the output there is no echo to cancel. That covers the built-in headphone jack, Bluetooth headsets and anything named like headphones (AirPods, Beats, buds, headset). There the helper leaves voice processing off (nothing ducked, the microphone unprocessed), and the listener still listens over the voice. `aec_on_headphones: true` keeps voice processing on everywhere.

**AirPods.** Nothing changes from half-duplex here. If AirPods are the default input, opening their microphone (which the listener does in either mode) switches them to the Bluetooth headset profile, mono and lower in quality, for as long as the listener runs. Music sounds worse then, as it does today. If the built-in microphone is the default input, AirPods stay in high quality. This hasn't been measured: no AirPods were connected while this was built.

**Barge-in** (`bargein.py`). While the voice plays through the helper, each 32 ms frame of cancelled audio counts as your speech when two things hold:

- its VAD probability is at least `barge_in_vad`;
- its level is at least `barge_in_min_db`, and no more than `barge_in_echo_margin` dB below what the voice is playing at that moment. What is left of the voice's echo follows the voice's own level; your speech doesn't.

When frames amounting to 0.2 s count within 0.3 s (`barge_in_speech`, `barge_in_within`), the voice is paused at once, before your words are known. When the utterance has been transcribed:

| What you said | What happens |
| --- | --- |
| A stop phrase (`barge_in_stop_phrases`: "stop", "that's enough", "thank you", "hold on", …, with or without the name) | The voice is silenced (the helper at once, then `bin/hal stop` for anything queued behind it). Your words go nowhere and aren't logged. |
| "Go on", "continue", "sorry, go on" (`barge_in_resume_phrases`) | The voice resumes where it was paused. |
| The voice's own words | Some echo got through, so it is dropped and the voice resumes. The words are compared, in order, with the sentences the voice was saying at that moment. A single stray word counts as noise. |
| Nothing, a noise, one word | The voice resumes. |
| Anything else | The voice is silenced, and what you said goes to the turn logic as a request without the name (`how` `barge-in`), exactly as if you had said it after the voice. |

Speech heard over the voice that didn't interrupt it (too quiet or too short) is taken only if it contains the name. The follow-up, armed and `during_turn` windows don't apply to it, because it may be residual echo.

**Going on later.** When the voice is cut off, the listener records where: the position, the sentence it fell in, and the text from the start of that sentence. It keeps this in memory and in `state/listen/interrupted.json`, which holds the voice's own text only. "Go on" or "continue" (with or without the name, not while a turn is held) within `resume_seconds` says the rest, starting again at the beginning of the interrupted sentence, unless the voice has spoken since. The worker that renders speech writes `<wav>.json` beside each WAV: its text and each sentence's start and end, taken from Piper's per-sentence chunks. The file is deleted with the WAV.

**What voice processing does to the rest of the system** (measured on this MacBook, see below):

- Other apps capturing the microphone at the same time get a much quieter signal: room noise 17 dB lower, the voice about 30 dB lower. So by default (`aec_yield_to_other_input`) the helper drops voice processing while any other app captures audio, for example a call or dictation, and the listener is half-duplex meanwhile. Voice processing returns 5 s after the other app stops, never while HAL is speaking. Apple's `corespeechd`, which starts capturing when voice processing does, doesn't count.
- Other audio, such as music, is ducked while voice processing runs. The helper asks for the least (`aec_ducking` `"min"`): `afplay` came out about 5–6 dB quieter relative to the helper's own voice. That is an estimate, because the microphone path changes too.
- Apple's automatic gain control is off (`aec_agc`), so levels stay comparable. Its noise suppression can't be turned off without turning off voice processing.
- The system's output and input volumes are unchanged. The orange microphone indicator behaves as before.
- Starting voice processing reconfigures the output device, and audio playing at that instant can stutter. It happens when the listener starts, on device changes, and when the helper yields or comes back.

**Fallback.** If the helper is missing, won't start, exits or stops sending audio, the listener captures with the microphone as before. That is half-duplex, logged with the reason, and the helper is tried again after 1, 2, 4 … up to 15 minutes. If voice processing is unavailable for the current devices and the output isn't headphones, the listener gates as in half-duplex until the devices change. The log and `hal-listen status` say which mode is in use and why.

### Typing into Claude Code: the pty proxy

`hal-listen pty -- claude …` (`ptyproxy.py`, standard library only) runs `claude` on a pseudo-terminal. It relays the real terminal in raw mode, byte for byte both ways, and passes on the following:

- **Window size.** Changes reach `claude` and it redraws.
- **Job control.** Ctrl-Z works. `claude` runs as a foreground job under a tiny session-leader process, as it would under a shell. If `claude` were the session leader itself, the kernel would discard its SIGTSTP.
- **Signals.** SIGTERM, SIGHUP, SIGINT, SIGQUIT and SIGUSR1/2 sent to the proxy go to `claude`. A closed terminal sends SIGHUP.
- **Exit status.** The proxy exits with `claude`'s code, or is killed by the same signal.
- **Fallbacks.** If the proxy can't start for any reason, or stdin isn't a terminal, `claude` runs directly.
- **Typing requests.** The proxy accepts requests on `state/listen/pty.sock` (mode 0600).
  - The text is typed, then Enter is sent 0.15 s later as a separate write. Claude Code takes a burst that contains Enter as a paste.
  - Keys you press during those 0.15 s are held and passed on just after the Enter.
  - Control characters and escape sequences in the text are removed.
  - Each request has an id, and an id is never typed twice, so a retry after a lost reply is safe.
- **One proxy at a time.** Only one proxy takes requests (`proxy.lock`). A second `hal claude` still works, but nothing is typed into it.

The proxy sets `HAL_LISTEN_PROXY=<its pid>` for `claude`. Hooks inherit that variable, so only that session's hooks update its state file. `bin/hal-listen hook` exits at once anywhere else (7 ms).

tmux isn't installed here, and the proxy needs no extra software.

### When typing is unsafe

`session.py` decides at the instant of typing, inside the proxy's single-threaded loop. Typing is refused in these cases:

- **No session state yet.** The session's hooks haven't reported (`SessionStart`). This also covers the workspace-trust dialog at startup, which comes before it. Requests are held.
- **A dialog is open.** HAL says it needs you at the keyboard, and requests are held. The dialog kinds are listed below.
- **Unsent text in the prompt box.** You typed since your last Enter, Ctrl-C or Ctrl-U. HAL says so, and requests are held. A draft untouched for 60 s is taken as abandoned. Stray keys don't count: a draft that is blank, no more than 3 characters (spaces aside), or up to 16 characters with no dictionary word in it ("jk", "asdf") is cleared with Ctrl-U and the request typed. Pasted text, and text of more than one line, always count. The log says when a draft was last typed, and when stray keys were cleared.
- **A key was pressed within the last second.** Requests are held quietly and retried.

A turn in progress is not a reason to refuse: Claude Code queues what is typed.

The hooks maintain `state/listen/session-<proxy pid>.json`:

```json
{"version": 1, "proxy_pid": 4242, "session_id": "…", "started": true, "turn": "busy",
 "dialogs": [{"kind": "permission", "tool": "Bash", "key": "3f2a9c…", "agent": null, "since": 1731700000.1, "enter": true}],
 "last_stop": 1731699990.0, "last_event": "PermissionRequest", "updated": 1731700000.1}
```

These hook events open dialogs:

| Kind | Opened by |
| --- | --- |
| permission | `PermissionRequest`, which fires the moment the prompt is shown. The `permission_prompt` notification only comes after about 6 s without typing. |
| question | `PreToolUse` for `AskUserQuestion` |
| plan | `PreToolUse` for `ExitPlanMode` |
| elicitation | `Elicitation` |
| notification | `Notification` of type `permission_prompt`, `elicitation_dialog`, `elicitation_url_dialog`, `agent_needs_input` or `quota_auto_resume_stale` |

These events close them:

- `PostToolUse`, `PostToolUseFailure`, `PermissionDenied` and `PostToolBatch` close the dialog for the same tool call. `PermissionRequest` carries no `tool_use_id`, so the match is on tool name and input.
- `ElicitationResult` closes an elicitation dialog.
- At a turn boundary everything closes: `UserPromptSubmit`, `Stop`, `StopFailure`, `SessionStart`, `SessionEnd`, or `Notification` `idle_prompt`.

The real keyboard also counts, as seen by the proxy. A manual "No" fires no hook at all, and an approved tool reports only when it finishes:

- Esc or Ctrl-C after a dialog appeared closes it, and every older one.
- Enter answers the oldest permission or notification dialog.
- Questions, plans and elicitations can take several Enters, so only Esc or their own hook event closes them.
- Dialogs older than 30 minutes are ignored, so a missed event can't wedge input.

## Privacy

- **Audio stays in memory.** Captured audio goes from frame queue to segmenter to the model's input and is never written to disk.
- **Speech-to-text is fully local.** It works in memory: parakeet-mlx's own `transcribe()` reads a file, so `asr.py` feeds the samples to the model directly.
- **Ignored utterances leave no words anywhere.** `logs.ignored()` and `logs.discarded()` take no text argument. Ignored words never appear in the log, `listener.json`, the status output or the hook log.
  - The foreground `run` shows them on the terminal, only when stdout is a terminal or `--show-ignored` is given.
- **Addressed utterances are logged with their words.** The hook log records event names and dialog kinds, not prompts or tool inputs.
- **Only addressed turns reach the judge.** The text sent to the front desk's Haiku is the text of a turn the listener would type into Claude Code: it started with the name (or in the armed, follow-up or `during_turn` window) and is still being held. Unaddressed speech is never sent, and the desk logs only the number of words judged. `listener.json` shows a held turn's number of fragments and verdict, never its words. With `turn_judge: false` nothing at all leaves the machine before typing.
- **Pausing closes the microphone**, which turns off the orange indicator.
- **Speech over the voice** (`duplex` `"aec"`): speech that interrupts the voice and isn't a stop or resume phrase or the voice's own echo is treated as addressed, so it is typed and logged like any request. Stop phrases, echo, noise and anything heard over the voice without interrupting it (unless it has the name) are never logged with their words. The helper sends audio only to the listener, through a pipe, and writes nothing. `interrupted.json` and the `<wav>.json` sidecars hold HAL's own words, never yours.
- **Labels, marks and the lexicon** apply only to the request being typed: nothing is computed for ignored speech, and the lexicon writes nothing. `lexicon.txt`, the folder names and the alias table are read locally and never leave the machine; only a correction made to a request (`Kolodzin [heard as "Culdran"]`) goes, as part of that request, to Claude. The turn judge gets the words as heard, uncorrected.
- **Tests enforce this.** `tests/test_listen_privacy.py` checks every file the listener writes.

## Settings

All optional, in a `"listener"` object in `config.json`. A missing key, or one of the wrong type, falls back to the default. `hal/config.py` ignores this block.

| Key | Default | Meaning |
| --- | --- | --- |
| `autostart` | `true` | `hal claude` starts the listener for its session. |
| `names` | see above | Variants matched anywhere. |
| `vocative_names` | see above | Variants matched only when calling: `cloud`, `clawed`, … |
| `lead_words` | `hey`, `good`, `morning`, `thanks`, … | Words allowed before a vocative variant without a comma. |
| `name_only_words` | `hey`, `ok`, `okay`, `oh`, `um`, … | With the name and nothing else, the utterance is "the name alone". |
| `name_only_reply` | `"Yes?"` | The answer to the name alone; for example `"Yes, Dr. Alexander?"`. |
| `armed_seconds` | `8` | No name needed for this long after that reply. |
| `name_only_grace` | `0.25` | Wait before answering, in case you go on. |
| `follow_up_seconds` | `0` | No name needed this long after a spoken reply. 0 is off. |
| `during_turn` | `true` | No name needed while Claude is still working on a request the listener typed. |
| `stop_phrases` | `stop listening`, `pause listening`, … | |
| `paused_reply`, `dialog_notice`, `draft_notice`, `unreachable_notice` | | What HAL says. |
| `hold_seconds` | `120` | A held request is dropped after this. |
| `keyboard_quiet` | `1.0` | No typing for this long after a keystroke. |
| `draft_hold_seconds` | `60` | Unsent text blocks typing for this long after the last key. |
| `draft_ignore_chars` / `draft_ignore_wordless` | `3` / `true` | A draft this short, or (when true) a short one with no dictionary word, is stray keys: cleared, not waited for. |
| `submit_delay` | `0.15` | Pause between the text and Enter. |
| `dialog_max_age` | `1800` | Older dialogs are ignored. |
| `require_session_state` | `true` | Don't type until the hooks have reported. |
| `vad_threshold` / `vad_neg_threshold` | `0.5` / `0.35` | |
| `end_silence` | `0.8` | Silence that ends an utterance (with turns, a fragment). |
| `turn_taking` | `"semantic"` | `"semantic"`: decide by what was said (Turns). `"silence"`: every utterance is a request. |
| `turn_end_silence` | `1.2` | A finished turn is sent after this much silence. |
| `turn_max_wait` | `60` | An unfinished turn is sent as it stands after this much silence. |
| `turn_unclear_wait` | `3` | An unclear turn with no judgement is sent after this much silence. |
| `turn_judge` / `turn_judge_timeout` | `true` / `2.5` | Ask the front desk's Haiku when the rules can't tell; give up after this long. |
| `turn_retranscribe` | `true` | Transcribe a turn of several fragments afresh as a whole. |
| `turn_send_phrases` / `turn_cancel_phrases` | see above | Send or discard a turn being held. |
| `turn_cancel_reply` | `""` | Said when a turn is cancelled. |
| `turn_cue` / `turn_cue_after` | `"none"` / `10` | `"none"`, `"sound"` or `"words"`, once per turn after this much silence. |
| `turn_cue_sound` / `turn_cue_words` | `Tink.aiff` / `"Go on."` | |
| `pre_roll` / `post_roll` | `0.4` / `0.2` | |
| `min_speech` / `max_utterance` | `0.25` / `30` | |
| `echo_tail` | `0.6` | Deaf for this long after HAL stops speaking (half-duplex). |
| `duplex` | `"half"` | `"half"`: nothing is heard while HAL speaks. `"aec"`: echo-cancelled capture through `duplex/hal-duplex`, so you can interrupt the voice (falls back to `"half"` by itself). |
| `aec_ducking` | `"min"` | How much other audio is ducked while voice processing runs: `"min"`, `"default"`, `"mid"`, `"max"`. |
| `aec_agc` | `false` | Apple's automatic gain control on the microphone. |
| `aec_on_headphones` | `false` | Keep voice processing on with headphones (there is no echo to cancel). |
| `aec_yield_to_other_input` | `true` | Drop voice processing while another app captures audio (a call); half-duplex meanwhile. |
| `barge_in` | `true` | With `"aec"`, speech over the voice interrupts it. `false`: it is heard but never interrupts. |
| `barge_in_vad` / `barge_in_speech` / `barge_in_within` | `0.6` / `0.2` / `0.3` | A frame is speech at this VAD probability; this many seconds of it within this window pause the voice. |
| `barge_in_min_db` / `barge_in_echo_margin` | `-50` / `20` | …and only frames at least this loud (dBFS, after cancellation) and no more than this many dB below what the voice is playing. |
| `barge_in_echo_ratio` | `0.6` | Share of the words heard, in order, that must be the voice's for it to count as echo. |
| `barge_in_stop_phrases` / `barge_in_resume_phrases` | see above | Silence the voice; resume it. |
| `resume_seconds` | `300` | "Go on" resumes a cut-off voice for this long (unless it has spoken since). |
| `spoken_label` | `true` | Start what is typed with `[spoken HH:MM]` and allow the marks below. `false`: typed exactly as before, with no label, marks, corrections or name aliases. |
| `lexicon` | `true` | Correct names from the lexicon, and mark each correction. |
| `lexicon_file` / `lexicon_dirs` | `lexicon.txt` / `[]` (for example `["~/Source", "~/Papers"]`) | The hand-edited list; folders whose sub-folder names are added. |
| `doubt_marks` / `doubt_threshold` | `true` / `0.82` | Mark words the transcriber was less sure of than this, at most two a sentence. |
| `pause_marks` / `pause_mark_seconds` | `true` / `3` | Mark pauses at least this long between the fragments of a turn. |
| `interruption_note` | `true` | Say in the label which sentence the voice was in when the request cut it off. |
| `device` | `null` | Input device: index or name substring. `null` is the system default. |
| `asr_model` / `vad_model` | `models/parakeet-tdt-0.6b-v2` / `models/silero_vad.onnx` | A project path, or a HuggingFace id for `asr_model`. |

`desk_turns` (default `true`) is a front-desk setting, at the top level of `config.json`: the desk keeps the third warm model for turn judgements.

## Measured (MacBook Pro M4 Max, macOS 15.6)

- **Latency.** From the end of speech to the line arriving in the session, in the end-to-end tests: 1.39–1.45 s for a request the rules find finished (the 1.2 s `turn_end_silence`, the 0.15 s pause before Enter, and a few milliseconds of socket and pty; speech-to-text, 60–100 ms, happens inside the silence); 2.15 s for one the real Haiku had to judge (0.8 s `end_silence`, then about 1.1 s for the judgement). Before turns it was 1.03–1.07 s with `end_silence` 0.8, and about 2.3 s with 2.0.
- **The judge** (real Claude Haiku through the front desk, 33 transcripts: the author's own fragments, 16 unfinished and 17 finished): Haiku judged all 33 correctly; the rules decided 17 of them (10 finished, 7 unfinished), all correctly, and left 16 to Haiku. The answer took 0.80–1.70 s, median 0.88 s. The set is small and was written for the purpose, so expect mistakes on real speech.
- **Model load.** The model loads in about 3 s from a warm disk cache, plus a warm-up. Each 32 ms frame costs 0.08 ms of VAD.
- **Hook cost.** The state hook takes about 40 ms per event in the proxied session and 7 ms in any other session.
- **Accuracy** (`tests/listen_accuracy.py`: 16 addressed and 18 unaddressed phrases spoken by 18 macOS voices, 612 utterances through VAD, Parakeet and the rule). The unaddressed phrases include "stored in the cloud", "clawed", "clause", "clothes", "cloudy", "Claudia" and "a clod":

  | Voices | Missed (addressed but ignored) | Accepted (not addressed) |
  | --- | --- | --- |
  | 6 natural voices (Samantha, Daniel, Karen, Moira, Rishi, Tessa) | 0 of 96 | 0 of 108 |
  | 12 robotic formant voices (Fred, Ralph, Albert, Junior, Eddy, Flo, …) | 38 of 192 (19.8%) | 0 of 216 |

  Most of the robotic misses aren't fixable by a word list. "Claude" came out as "Lord", "Globe", "Float" or "Flawed", was dropped altogether, or became "cloth" with no comma before it.

- **Duplex** (`duplex` `"aec"`, built-in microphone and speakers, system output volume 94 %, a quiet room; levels in dBFS over 32 ms frames, measured with `hal-duplex` and a plain capture running side by side):

  | | Median | 90th percentile | Peak | VAD speech | Words of the sentence transcribed |
  | --- | --- | --- | --- | --- | --- |
  | HAL's voice at the microphone, no voice processing (2.6 s sentence) | −23.3 | −17.8 | −16.0 | 2.2 s | 9 of 9 |
  | The same after cancellation, played through the helper | −92.7 | −61.4 | −46.2 | 0.16 s | 0 (one stray word) |
  | The same after cancellation, played by `afplay` | −67.8 (the room's floor) | −54.0 | −43.8 | 0 s | 0 |
  | 5 s of the voice through the helper | −70.0 | −51.6 | −24.6 | 0.32 s | (no false barge-in) |

  - **Detection.** Speech mixed into those cancelled frames was detected after 0.21–0.28 s at −25 to −35 dBFS, and after 0.3–1.2 s at −40 dBFS; a short phrase at −40 dBFS was sometimes missed.
  - **Stopping.** The helper acknowledged a pause in 3 ms, and the voice was gone from the microphone 0.19 s after the command.
  - **From the start of speech to silence:** about 0.45–0.5 s. Lower `barge_in_speech` for faster, at more risk of the voice pausing on its own echo.
  - **Cost.** The helper uses about 10 % of one core (peaks 15 %) and 110 MB.
  - **Double-talk wasn't measurable here.** A second synthetic voice played from the same speakers is cancelled too, as other audio. Real speech from the room is not in the canceller's reference, but whether voice processing suppresses it while HAL speaks has to be tried by a person.

## Testing

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_listen*.py'         # about 25 s
HAL_LISTEN_E2E=1 .venv/bin/python -m unittest tests.test_listen_e2e -v       # about 3.5 minutes
.venv-listen/bin/python tests/listen_accuracy.py                             # several minutes
```

| File | Covers |
| --- | --- |
| `test_listen_addressing.py` | The addressing rule on a corpus of addressed and unaddressed transcripts, variants, name alone, stop listening, and the decider (arming, follow-up window, long utterances). |
| `test_listen_turns.py` | Turns: the rules, the spoken controls, the turn-taker with a stand-in judge (complete, unfinished, unavailable, too slow, maximum wait, cue), the listening loop (a request across pauses of 3, 10 and 30 s, name alone, stop listening, the gate, the follow-up window, `during_turn`, silence mode), and the client for the desk's `turn` request against a stand-in socket. |
| `test_listen_segmenter.py` | The segmenter on scripted probabilities, and Silero on `say` speech: one utterance, silence and noise, two utterances close together and further apart, a long utterance. |
| `test_listen_gate.py` | The voice gate with fake PID files, the listening loop dropping an interrupted utterance, and the gate against a real `bin/hal say` worker (silent fake player). |
| `test_listen_duplex.py` | Duplex `"aec"`: stop and resume phrases, the echo check, sentence positions, the barge-in detector, the listening loop over an in-memory duplex source (stop phrase, echo, noise, request, "go on", gating when cancellation is unavailable, no windows for quiet overlap), `DuplexSource` against `tests/duplex_standin.py` (voice processing on or off, headphones, a device change to AirPods, a missing, failing or dying helper, retrying), the speaker's sentence sidecars and player choice, the settings, and the Swift client's protocol against a stand-in socket (silent). |
| `test_listen_session.py` | The state reducer for idle, busy and dialog events, keyboard handling (including the kitty keyboard protocol), the verdict, and the hook command (silent, fast, never failing). |
| `test_listen_injector.py` | Holding, announcing, retrying, expiring, and ids. |
| `test_listen_privacy.py` | No ignored words in any file, and no audio files. |
| `test_listen_spoken.py` | The label and marks: format, unforgeable, `plain()`, never spoken; every hook and front-desk check with and without the label (pure functions, the real hook command, the desk with its stand-in model); the lexicon (file, folders, reloading, pronunciation keys, every observed mistake, sound-alike corrections and flags, ordinary look-alike words left alone, the name's aliases only when calling); doubtful words, pauses, the interruption, each setting off; the listening loop (label time, pauses, corrections, silence mode, privacy); and, with `.venv-listen` and the model, Parakeet's confidences on synthesized clear and hard speech and `hal-listen run --file` on a synthesized recording. |
| `test_listen_ptyproxy.py` | A scripted terminal around a stand-in program and around a real `bash`: byte-exact keys, output, resizing, exit codes and signals, Ctrl-Z and `fg`, typing exactly once, refusals, keys held during typing, one proxy at a time. |
| `test_listen_lifecycle.py` | Autostart, status, one listener at a time, pause and resume, stop, and exiting with the session (even a killed one). |
| `test_listen_e2e.py` | Synthesized audio through the whole pipeline into a stand-in `claude` that fires the real HAL hooks (silent player, temporary state). `TurnsEndToEndTest` adds a front desk and one request spoken in four pieces with pauses of 3, 10 and 30 s; with `HAL_LISTEN_E2E_HAIKU=1` its desk uses the real Haiku (a few short calls). |

Every test uses a temporary `HAL_STATE_DIR` and the silent `tests/fake_player.py`. None opens the microphone.

Testing aids for the background listener: `HAL_LISTEN_FILE` makes it hear an audio file instead of the microphone, and `HAL_LISTEN_FAKE_ASR` makes it read transcripts from a file, one per line, instead of running the model. `hal-listen run --file F` does the same in the foreground.

## Known limits

- **Turns read the transcript.** The rules and the judge see Parakeet's words and punctuation, not the voice. With one synthetic voice (Samantha), Parakeet wrote a final "… my paper and" as "… my paper Anne?", a finished question, and the request went too early. A request that is finished as far as it goes ("Claude, open my thesis.") is sent after 1.2 s of silence even if you meant to go on; what you add after the acknowledgement has started is lost to the voice gate. Say "that's all" or "go ahead" if a finished request is held, and expect a minute's wait without it if the judge wrongly thinks you are mid-thought.
- **Anyone's speech can join a turn.** While a turn is held, and while Claude works on a spoken request (`during_turn`), speech needs no name, so someone else talking in the room is typed as part of it (and judged). The same is true of the follow-up window.
- **Third-person mentions count.** "I asked Claude yesterday" contains the name and is typed. To require the calling position for "Claude" too, move it from `names` to `vocative_names`.
- **Half-duplex by default.** With `duplex` `"half"` you can't interrupt HAL by voice, and anything said while HAL speaks is lost. With `"aec"` you can (above), except when it falls back: while another app captures audio, while voice processing is unavailable for the current devices, or while HAL's voice is being prepared and the helper isn't running.
- **Interrupting is tuned on synthetic speech.** Real double-talk (you speaking over HAL through the laptop speakers) hasn't been measured, and nor have AirPods or wired headphones. A loud room or music may pause the voice, which then resumes once the words show it was nothing. A quiet voice (about −40 dBFS at the microphone) may not interrupt it. If you repeat the voice's own words back to it, that counts as echo.
- **A sandboxed command's network-request prompt fires no `PermissionRequest`.** Only the `permission_prompt` notification marks it, about 6 s late, so a request spoken in those first seconds could be typed into it.
- **The draft is modelled from keys.** Text recalled with the up arrow, or inserted by `/voice`, isn't seen, and editing after moving the cursor is taken as typing at the end. Backspace, Ctrl-W and Alt-Backspace delete from the end. Stray keys are judged against `/usr/share/dict/words`, so a short draft of names or terms the dictionary lacks ("jmacs") counts as stray and is cleared.
- **Claude Code's vim mode isn't supported.** In NORMAL mode the typed characters would be commands.
- **Tested only on synthetic speech.** Parakeet does well on the natural macOS voices and worse on robotic ones. Real rooms (distance, noise, other voices, music) haven't been measured.
- **The proxy has been run around the real `claude`, but not logged in.** The tests used a throwaway config, so typing into the real prompt box hasn't been seen.

## Integration (changes to files the listener doesn't own)

### 1. `bin/hal`: launch through the proxy

Replace the `claude` branch with:

```bash
# `hal claude [args]`: start Claude Code as the voice ("master") session, under
# the listener's pty proxy so spoken requests can be typed into it (LISTENER.md).
# HAL_LISTEN=0 skips the proxy; so does a missing bin/hal-listen.
if [ "$1" = claude ]; then
  shift
  if [ "${HAL_LISTEN:-1}" != 0 ] && [ -x "$root/bin/hal-listen" ]; then
    HAL_VOICE=1 exec "$root/bin/hal-listen" pty -- claude "$@"
  fi
  HAL_VOICE=1 exec claude "$@"
fi
```

### 2. `.claude/settings.json`: the state hook

Add these to `"hooks"`. For `UserPromptSubmit` and `Stop`, which already have HAL's entries, append the new group to the existing array. The per-tool events are `async`: they only close dialogs, matched to their own tool call, so arriving late is harmless. Everything else is synchronous, so the state is current before Claude Code moves on. Each run costs about 40 ms.

```json
"SessionStart":       [{"hooks": [{"type": "command", "command": "\"$CLAUDE_PROJECT_DIR\"/bin/hal-listen hook", "timeout": 5}]}],
"SessionEnd":         [{"hooks": [{"type": "command", "command": "\"$CLAUDE_PROJECT_DIR\"/bin/hal-listen hook", "timeout": 5}]}],
"UserPromptSubmit":   [ /* HAL's existing group */, {"hooks": [{"type": "command", "command": "\"$CLAUDE_PROJECT_DIR\"/bin/hal-listen hook", "timeout": 5}]}],
"Stop":               [ /* HAL's existing group */, {"hooks": [{"type": "command", "command": "\"$CLAUDE_PROJECT_DIR\"/bin/hal-listen hook", "timeout": 5}]}],
"StopFailure":        [{"hooks": [{"type": "command", "command": "\"$CLAUDE_PROJECT_DIR\"/bin/hal-listen hook", "timeout": 5}]}],
"PermissionRequest":  [{"hooks": [{"type": "command", "command": "\"$CLAUDE_PROJECT_DIR\"/bin/hal-listen hook", "timeout": 5}]}],
"PreToolUse":         [{"matcher": "AskUserQuestion|ExitPlanMode",
                        "hooks": [{"type": "command", "command": "\"$CLAUDE_PROJECT_DIR\"/bin/hal-listen hook", "timeout": 5}]}],
"Notification":       [{"hooks": [{"type": "command", "command": "\"$CLAUDE_PROJECT_DIR\"/bin/hal-listen hook", "timeout": 5}]}],
"Elicitation":        [{"hooks": [{"type": "command", "command": "\"$CLAUDE_PROJECT_DIR\"/bin/hal-listen hook", "timeout": 5}]}],
"ElicitationResult":  [{"hooks": [{"type": "command", "command": "\"$CLAUDE_PROJECT_DIR\"/bin/hal-listen hook", "timeout": 5}]}],
"PostToolUse":        [{"hooks": [{"type": "command", "command": "\"$CLAUDE_PROJECT_DIR\"/bin/hal-listen hook", "async": true}]}],
"PostToolUseFailure": [{"hooks": [{"type": "command", "command": "\"$CLAUDE_PROJECT_DIR\"/bin/hal-listen hook", "async": true}]}],
"PostToolBatch":      [{"hooks": [{"type": "command", "command": "\"$CLAUDE_PROJECT_DIR\"/bin/hal-listen hook", "async": true}]}],
"PermissionDenied":   [{"hooks": [{"type": "command", "command": "\"$CLAUDE_PROJECT_DIR\"/bin/hal-listen hook", "async": true}]}]
```

The hook prints nothing and always exits 0, so it never decides anything for Claude Code: dialogs show as before.

### 3. `install.sh` (only for use in other projects)

The proxy works in any directory, but typing needs the hooks above to run in that session. To use `hal claude` with the listener outside this project, have `install.sh` add the same entries to `~/.claude/settings.json`, with the absolute path `<project>/bin/hal-listen hook`. The hook does nothing in sessions not started under the proxy, so registering it globally is harmless.

### 4. `config.json` (optional)

Add a `"listener": { … }` block only to change a default, for example `"listener": {"follow_up_seconds": 5}`.

### 5. `README.md`

A short section, for example:

> ### Always listening
>
> Instead of `/voice`, the laptop microphone can listen all the time: say "Claude, …" and the request is typed into the session started with `hal claude`; everything else is ignored, and nothing leaves the machine. Run `scripts/setup-listen.sh` once; `hal-listen status` shows what it is doing and `hal-listen pause` stops it. See LISTENER.md.

In "Known limits", the line "HAL keeps talking while you dictate … use headphones" applies only to `/voice`: the listener doesn't listen while HAL speaks.

### 6. The voice (for duplex "aec")

`hal/speaker.py` writes `<wav>.json` beside each WAV it renders: its text and its sentences' times (`synthesize(..., sidecar=True)`). It deletes the file with the WAV. `hal/desk.py` asks for the same. `hal/config.py`'s `player_command()` plays through `duplex/hal-duplex play` while the helper's socket exists; `HAL_PLAYER` still wins. With `duplex` `"half"` the socket never exists, so the voice plays exactly as before.

### 7. Nothing else

`hal/cli.py` needs no changes. `hal/hook.py`, `hal/desk.py` and `hal/textclean.py` read past the listener's label and marks through `hal/spoken.py` (see [What the model is told](#what-the-model-is-told)). The listener reads `speaker.PID_FILE` and `speaker.ACK_PID_FILE`, speaks through `bin/hal say --raw --wait` and stops the voice with `bin/hal stop`. If those names or commands change, `settings.speaker_pid_files()`, `service.VoiceOut` and `service.stop_hal_voice()` are where to follow them.
