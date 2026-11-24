#!/bin/bash
# The HAL voice session window. At login, launchd/local.hal.voice-session.plist
# (through scripts/open-voice-window.sh) opens this file in iTerm2, or Terminal,
# which runs it in a new window. See AUTOSTART.md.
#
# It runs the supervised voice session (bin/hal-session), then leaves a shell
# open in the project folder, so its last message stays on screen and
# `bin/hal-session` starts the session again. Closing the window ends it all.
cd "$(dirname "$0")/.." || exit 1
bin/hal-session
exec "${SHELL:-/bin/zsh}" -l
