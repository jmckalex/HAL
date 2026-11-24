"""HAL 9000 voice for Claude Code: speak Claude's replies with a Piper voice.

Modules:
    textclean  Markdown -> speakable prose, plus the length cap (pure, stdlib only)
    speaker    synthesise with Piper and play, one utterance at a time
    hook       Claude Code Stop / UserPromptSubmit hook handler
    desk       the front desk: a long-lived process with a warm small model
               and the voice loaded, which hears each prompt first
    summarize  spoken summaries of long replies
    config     paths, config.json, mute flag, log
    cli        the `hal` command
"""
