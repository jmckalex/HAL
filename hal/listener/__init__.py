"""Always-listening front end for the HAL voice (see LISTENER.md).

The microphone is transcribed locally; an utterance addressed to "Claude" is
typed into the interactive Claude Code session, which runs under a small
pseudo-terminal proxy. Nothing here changes how the existing hooks speak.

Modules (stdlib-only unless noted, so the hook and the proxy start fast):

    settings    paths under state/listen/, the "listener" block of config.json
    addressing  is this transcript addressed to Claude? (pure text rules)
    session     session-state file kept by hooks; may text be typed now?
    hookstate   `hal-listen hook`: Claude Code hook entry point for that file
    ptyproxy    runs `claude` under a pty, relays the terminal, types on request
    control     client for the proxy's Unix socket
    gate        half-duplex gate: discard audio while the voice is speaking
    injector    hand addressed text to the proxy; hold it while a dialog is open
    decider     addressing + "Yes?" arming + follow-up window
    logs        listen.log, which never records unaddressed text
    segmenter   VAD-driven utterance segmentation (numpy)
    vad         Silero VAD through onnxruntime (numpy, onnxruntime)
    asr         Parakeet speech-to-text through MLX (parakeet-mlx)
    audio       microphone and file sources (sounddevice for the microphone)
    service     the listening loop that ties these together
    cli         the `hal-listen` command

Keep this file free of imports: the hook path imports the package.
"""
