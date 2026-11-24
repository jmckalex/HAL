#!/bin/bash
# Turn off the HAL autostart: unload the LaunchAgents that
# scripts/install-autostart.sh installed and remove their files from
# ~/Library/LaunchAgents. See AUTOSTART.md.
#
#   scripts/uninstall-autostart.sh [--dry-run]
#
# The menu-bar item quits. A voice session that is running keeps running: it
# belongs to its terminal window, not to launchd. The project's own files
# (launchd/, menubar/HALMenu, state/) are left as they are.
#
# Safe to run again: what is already gone is skipped. --dry-run prints what
# would be done and changes nothing.
#
# Testing aids (environment): HAL_LAUNCH_AGENTS_DIR stands in for
# ~/Library/LaunchAgents, and HAL_LAUNCHCTL for /bin/launchctl. With a stand-in
# folder and no HAL_LAUNCHCTL, launchctl is never run.
set -euo pipefail

dry=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) dry=1 ;;
    -h|--help) awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"; exit 0 ;;
    *) echo "uninstall-autostart: unknown option $arg (see --help)" >&2; exit 2 ;;
  esac
done

real_agents="$HOME/Library/LaunchAgents"
agents="${HAL_LAUNCH_AGENTS_DIR:-$real_agents}"
if [ -n "${HAL_LAUNCHCTL:-}" ]; then
  launchctl="$HAL_LAUNCHCTL"
elif [ "$agents" = "$real_agents" ]; then
  launchctl=/bin/launchctl
else
  launchctl=""
fi
domain="gui/$(id -u)"

quoted() {
  local out="" a
  for a in "$@"; do out="$out $(printf '%q' "$a")"; done
  printf '%s' "${out# }"
}
act() {
  if [ "$dry" = 1 ]; then echo "  would run: $(quoted "$@")"; else echo "  run: $(quoted "$@")"; "$@"; fi
}
lctl() {
  if [ -z "$launchctl" ]; then
    echo "  skipped (stand-in folder, so no launchctl): launchctl $(quoted "$@")"
    return 0
  fi
  act "$launchctl" "$@"
}
is_loaded() {
  [ -n "$launchctl" ] && "$launchctl" print "$domain/$1" >/dev/null 2>&1
}

[ "$dry" = 1 ] && echo "Dry run: nothing will be changed."
echo "LaunchAgents: $agents"

for label in local.hal.menubar local.hal.voice-session; do
  dest="$agents/$label.plist"
  echo "$label:"
  if is_loaded "$label"; then
    lctl bootout "$domain/$label"
  else
    echo "  not loaded"
  fi
  if [ -f "$dest" ]; then
    if [ "$(plutil -extract Label raw -o - "$dest" 2>/dev/null)" = "$label" ]; then
      act rm -f "$dest"
    else
      echo "  $dest does not have the Label $label; left alone"
    fi
  else
    echo "  no $dest"
  fi
done

if [ "$dry" = 1 ]; then
  echo "Dry run finished: nothing was changed."
else
  echo "Done. Nothing HAL-related starts at login now; a running voice session is unaffected."
  echo "Turn it back on with scripts/install-autostart.sh."
fi
