#!/bin/bash
# Turn on the HAL autostart: install the LaunchAgents in launchd/ into
# ~/Library/LaunchAgents and load them. See AUTOSTART.md. The plists there are
# templates: @HAL_ROOT@ is replaced with this project's folder in the copies.
#
#   scripts/install-autostart.sh [--dry-run] [--no-menubar]
#
#   local.hal.voice-session  at login, opens an iTerm2 (or Terminal) window in
#                            this folder running the voice session, kept up by
#                            bin/hal-session
#   local.hal.menubar        the menu-bar item, menubar/HALMenu (built here
#                            first if it is missing or older than its source)
#
# Loading local.hal.voice-session runs it once at once: it opens a window only
# if no voice session is running. Loading local.hal.menubar starts the item.
#
# Safe to run again: an agent that is already installed and loaded is left
# alone; one whose file has changed is replaced and reloaded.
#   --dry-run     print what would be done; change nothing
#   --no-menubar  leave the menu-bar item out
# Undo with scripts/uninstall-autostart.sh.
#
# Testing aids (environment): HAL_LAUNCH_AGENTS_DIR stands in for
# ~/Library/LaunchAgents, and HAL_LAUNCHCTL for /bin/launchctl. With a stand-in
# folder and no HAL_LAUNCHCTL, launchctl is never run.
set -euo pipefail

root="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dry=0
menubar=1
for arg in "$@"; do
  case "$arg" in
    --dry-run) dry=1 ;;
    --no-menubar) menubar=0 ;;
    -h|--help) awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"; exit 0 ;;
    *) echo "install-autostart: unknown option $arg (see --help)" >&2; exit 2 ;;
  esac
done

real_agents="$HOME/Library/LaunchAgents"
agents="${HAL_LAUNCH_AGENTS_DIR:-$real_agents}"
if [ -n "${HAL_LAUNCHCTL:-}" ]; then
  launchctl="$HAL_LAUNCHCTL"
elif [ "$agents" = "$real_agents" ]; then
  launchctl=/bin/launchctl
else
  launchctl=""   # a stand-in folder: never load anything for real
fi
domain="gui/$(id -u)"

quoted() {
  local out="" a
  for a in "$@"; do out="$out $(printf '%q' "$a")"; done
  printf '%s' "${out# }"
}
# Run a command and say so, or with --dry-run only say what it would be.
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
is_loaded() {   # read-only: launchctl print
  [ -n "$launchctl" ] && "$launchctl" print "$domain/$1" >/dev/null 2>&1
}

[ "$dry" = 1 ] && echo "Dry run: nothing will be changed."
echo "Project:      $root"
echo "LaunchAgents: $agents"

labels=(local.hal.voice-session)
[ "$menubar" = 1 ] && labels+=(local.hal.menubar)

# Fill in the templates: @HAL_ROOT@ -> this folder.
rendered="$(mktemp -d "${TMPDIR:-/tmp}/hal-autostart.XXXXXX")"
trap 'rm -rf "$rendered"' EXIT
root_sed="$(printf '%s' "$root" | sed 's/[\\&|]/\\&/g')"
for label in "${labels[@]}"; do
  sed "s|@HAL_ROOT@|$root_sed|g" "$root/launchd/$label.plist" > "$rendered/$label.plist"
done

# Check the plists before touching anything.
for label in "${labels[@]}"; do
  src="$rendered/$label.plist"
  if ! plutil -lint -s "$src"; then
    echo "install-autostart: launchd/$label.plist is not a valid property list" >&2; exit 1
  fi
  if [ "$(plutil -extract Label raw -o - "$src")" != "$label" ]; then
    echo "install-autostart: launchd/$label.plist does not have the Label $label" >&2; exit 1
  fi
  prog="$(plutil -extract ProgramArguments.0 raw -o - "$src")"
  case "$prog" in
    "$root"/*) ;;
    *) echo "install-autostart: launchd/$label.plist runs $prog, which is outside this project ($root)." >&2
       echo "Its paths should start with @HAL_ROOT@." >&2
       exit 1 ;;
  esac
  if [ "$label" = local.hal.voice-session ] && [ ! -x "$prog" ]; then
    echo "install-autostart: $prog is missing or not executable" >&2; exit 1
  fi
done

# The menu-bar item has to be built before launchd can start it.
if [ "$menubar" = 1 ]; then
  bin="$root/menubar/HALMenu"
  if [ ! -x "$bin" ] || [ "$root/menubar/HALMenu.swift" -nt "$bin" ]; then
    echo "Menu-bar item: building menubar/HALMenu"
    act "$root/menubar/build.sh"
  else
    echo "Menu-bar item: menubar/HALMenu is up to date"
  fi
fi

# launchd writes the agents' output to state/; it must exist.
if [ ! -d "$root/state" ]; then
  echo "State folder: creating $root/state"
  act mkdir -p "$root/state"
fi

if [ ! -d "$agents" ]; then
  echo "Creating $agents"
  act mkdir -p "$agents"
fi

for label in "${labels[@]}"; do
  src="$rendered/$label.plist"
  dest="$agents/$label.plist"
  echo "$label:"
  changed=1
  if [ -f "$dest" ] && cmp -s "$src" "$dest"; then
    changed=0
    echo "  $dest is up to date"
  elif [ -e "$dest" ]; then
    echo "  $dest differs from launchd/$label.plist: replace it"
    act cp "$src" "$dest"
  else
    echo "  not installed: copy launchd/$label.plist there"
    act cp "$src" "$dest"
  fi
  if is_loaded "$label"; then
    if [ "$changed" = 0 ]; then
      echo "  already loaded"
      continue
    fi
    echo "  loaded from the old file: reloading"
    lctl bootout "$domain/$label"
    if [ "$dry" = 0 ] && [ -n "$launchctl" ]; then
      for _ in 1 2 3 4 5 6 7 8 9 10; do is_loaded "$label" || break; sleep 0.5; done
    fi
  fi
  lctl enable "$domain/$label"   # in case it was disabled before; harmless otherwise
  lctl bootstrap "$domain" "$dest"
  if [ "$label" = local.hal.voice-session ]; then
    echo "  (loading runs it once now: it opens a window only if no voice session is running)"
  fi
done

if [ "$dry" = 1 ]; then
  echo "Dry run finished: nothing was changed."
else
  echo "Done. At the next login the voice session opens in its own window."
  echo "macOS may show a \"Background Items Added\" notice; the items are listed under"
  echo "System Settings > General > Login Items & Extensions. Undo: scripts/uninstall-autostart.sh"
fi
