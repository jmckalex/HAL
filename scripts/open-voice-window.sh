#!/bin/bash
# Open a terminal window in the project folder that runs the HAL voice session
# (launchd/hal-voice.command, which runs bin/hal-session). See AUTOSTART.md.
#
#   scripts/open-voice-window.sh [--login] [--dry-run]
#
# Used by the login agent (launchd/local.hal.voice-session.plist, with --login)
# and by the menu-bar item's "Start the voice session".
#
# - Does nothing if a voice session is already running (bin/hal-session --check).
# - Opens the window with `open`, in iTerm2 if it is installed, otherwise in
#   Terminal. No AppleScript, so no Automation permission is needed. iTerm2 may
#   ask once to confirm running the file.
# - --login first waits for the desktop (the Dock, up to 60 s), then up to 60 s
#   for a network route, so the session doesn't start before Wi-Fi is up. It
#   opens the window anyway when that time is up.
# - --dry-run prints what it would do, waits for nothing and opens nothing.
#
# Testing aids (environment): HAL_OPEN replaces /usr/bin/open; HAL_TERMINAL_APP
# set to "iterm" or "terminal" skips the check for iTerm2; HAL_STATE_DIR as for
# the other tools.

src="${BASH_SOURCE[0]}"
while [ -L "$src" ]; do
  dir="$(cd -P "$(dirname "$src")" && pwd)"
  src="$(readlink "$src")"
  [[ "$src" == /* ]] || src="$dir/$src"
done
root="$(cd -P "$(dirname "$src")/.." 2>/dev/null && pwd)"
command_file="$root/launchd/hal-voice.command"
open_cmd="${HAL_OPEN:-/usr/bin/open}"
log_file="${HAL_STATE_DIR:-$root/state}/autostart/autostart.log"

login=0
dry=0
for arg in "$@"; do
  case "$arg" in
    --login) login=1 ;;
    --dry-run) dry=1 ;;
    -h|--help) awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"; exit 0 ;;
    *) echo "open-voice-window: unknown option $arg" >&2; exit 2 ;;
  esac
done

say() {
  echo "open-voice-window: $*"
  if [ "$dry" = 0 ]; then
    mkdir -p "$(dirname "$log_file")" 2>/dev/null
    printf '%s [%s] window: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$$" "$*" >>"$log_file" 2>/dev/null
  fi
}

quoted() {
  local out="" a
  for a in "$@"; do out="$out $(printf '%q' "$a")"; done
  printf '%s' "${out# }"
}

# Run a command, or with --dry-run print it.
run() {
  if [ "$dry" = 1 ]; then
    echo "open-voice-window: would run: $(quoted "$@")"
    return 0
  fi
  "$@"
}

iterm_installed() {
  case "${HAL_TERMINAL_APP:-}" in
    iterm) return 0 ;;
    terminal) return 1 ;;
  esac
  local d
  for d in /Applications/iTerm.app "$HOME/Applications/iTerm.app"; do
    [ -d "$d" ] && return 0
  done
  [ -n "$(/usr/bin/mdfind "kMDItemCFBundleIdentifier == 'com.googlecode.iterm2'" 2>/dev/null | head -n 1)" ]
}

if [ ! -x "$command_file" ]; then
  say "cannot open a window: $command_file is missing or not executable"
  exit 1
fi

if [ "$login" = 1 ]; then
  if [ "$dry" = 1 ]; then
    echo "open-voice-window: would wait for the Dock (up to 60 s), then for a network route (up to 60 s)"
  else
    waited=0
    while ! /usr/bin/pgrep -x Dock >/dev/null 2>&1 && [ "$waited" -lt 60 ]; do
      sleep 1; waited=$((waited + 1))
    done
    sleep 3   # let the rest of the login settle (restored apps, the menu bar)
    net=0
    while ! /sbin/route -n get default >/dev/null 2>&1 && [ "$net" -lt 60 ]; do
      sleep 2; net=$((net + 2))
    done
    if /sbin/route -n get default >/dev/null 2>&1; then
      say "login: waited ${waited}s for the desktop and ${net}s for the network"
    else
      say "login: waited ${waited}s for the desktop; still no network after ${net}s, opening the window anyway"
    fi
  fi
fi

if running="$("$root/bin/hal-session" --check 2>/dev/null)"; then
  say "$running; not opening another window"
  exit 0
fi

if iterm_installed; then
  if run "$open_cmd" -b com.googlecode.iterm2 "$command_file"; then
    [ "$dry" = 1 ] || say "opened an iTerm2 window running launchd/hal-voice.command"
    exit 0
  fi
  say "iTerm2 did not open the window; trying Terminal"
fi
if run "$open_cmd" -b com.apple.Terminal "$command_file"; then
  [ "$dry" = 1 ] || say "opened a Terminal window running launchd/hal-voice.command"
  exit 0
fi
say "could not open a terminal window"
exit 1
