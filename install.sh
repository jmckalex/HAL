#!/bin/bash
# Register (or remove) the HAL voice hooks in your user-level Claude Code
# settings, so HAL speaks in every project, not just this one.
#
#   ./install.sh                  add the hooks to ~/.claude/settings.json
#   ./install.sh --uninstall      remove them again
#   ./install.sh --dry-run        show what would change, write nothing
#   ./install.sh --settings FILE  operate on FILE instead (e.g. a test copy)
#
# Merges rather than overwrites: other settings and other hooks are left as
# they are (the file is re-indented with 2 spaces). The previous file is saved
# next to it as settings.json.bak-hal-<timestamp> before any change. Running
# it twice changes nothing the second time. If settings.json is a symlink,
# the file it points to is updated and the link is kept.
set -euo pipefail

root="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
settings="$HOME/.claude/settings.json"
mode=install
dry_run=0

while [ $# -gt 0 ]; do
  case "$1" in
    --uninstall) mode=uninstall ;;
    --dry-run) dry_run=1 ;;
    --settings) settings="${2:?--settings needs a file}"; shift ;;
    -h|--help) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "install.sh: unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

if [ "$mode" = install ] && [ ! -x "$root/.venv/bin/python" ]; then
  echo "install.sh: run $root/setup.sh first (no .venv yet)" >&2
  exit 1
fi
python="$root/.venv/bin/python"
[ -x "$python" ] || python="$(command -v python3)"

"$python" -I - "$mode" "$settings" "$root" "$dry_run" <<'PY'
import datetime, json, os, re, shlex, shutil, sys, tempfile

mode, settings_path, root, dry_run = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] == "1"
EVENTS = ("Stop", "UserPromptSubmit", "PreToolUse")
ASYNC = ("Stop", "PreToolUse")  # never wait on these
COMMAND = f"{shlex.quote(os.path.join(root, 'bin', 'hal'))} hook"
HAL_COMMAND = re.compile(r"""bin/hal['"]?\s+hook\s*$""")  # any HAL install, wherever it lives

def is_hal(handler):
    return (isinstance(handler, dict) and handler.get("type") == "command"
            and isinstance(handler.get("command"), str) and bool(HAL_COMMAND.search(handler["command"])))

target = os.path.realpath(os.path.expanduser(settings_path))
try:
    with open(target, encoding="utf-8") as f:
        original_text = f.read()
except FileNotFoundError:
    original_text = None
try:
    data = json.loads(original_text) if original_text and original_text.strip() else {}
except ValueError as exc:
    sys.exit(f"install.sh: {target} is not valid JSON ({exc}); nothing changed")
if not isinstance(data, dict):
    sys.exit(f"install.sh: {target} does not hold a JSON object; nothing changed")
hooks = data.get("hooks", {})
if not isinstance(hooks, dict):
    sys.exit(f"install.sh: \"hooks\" in {target} is not an object; nothing changed")

changes = []

def remove_hal(event):
    """Drop HAL handlers from one event; prune groups/events left empty."""
    groups = hooks.get(event)
    if not isinstance(groups, list):
        return []
    removed, kept_groups = [], []
    for group in groups:
        handlers = group.get("hooks") if isinstance(group, dict) else None
        if not isinstance(handlers, list):
            kept_groups.append(group)
            continue
        keep = [h for h in handlers if not is_hal(h)]
        removed += [h["command"] for h in handlers if is_hal(h)]
        if keep:
            kept_groups.append({**group, "hooks": keep})
    if kept_groups:
        hooks[event] = kept_groups
    else:
        hooks.pop(event, None)
    return removed

for event in EVENTS:
    # Stop and PreToolUse (speaking while Claude works) run in the background.
    # UserPromptSubmit is synchronous (it is quick, and its stdout tells Claude
    # what was spoken), with a short timeout.
    wanted = {"type": "command", "command": COMMAND}
    wanted.update({"async": True} if event in ASYNC else {"timeout": 5})
    existing = [h for g in hooks.get(event, []) if isinstance(g, dict) and isinstance(g.get("hooks"), list)
                for h in g["hooks"] if is_hal(h)]
    if mode == "install" and existing == [wanted]:
        continue  # already exactly as wanted
    removed = remove_hal(event)
    for cmd in removed:
        changes.append(f"- {event}: {cmd}")
    if mode == "install":
        hooks.setdefault(event, []).append({"hooks": [wanted]})
        changes.append(f"+ {event}: {COMMAND} ({'async' if event in ASYNC else 'sync, 5 s timeout'})")

if not changes:
    print(f"{target}: HAL hooks already {'installed' if mode == 'install' else 'absent'}; nothing to do")
    sys.exit(0)

if hooks:
    data["hooks"] = hooks
else:
    data.pop("hooks", None)
new_text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"

print(f"{target}:")
for line in changes:
    print(f"  {line}")
if data.get("disableAllHooks") is True:
    print("  note: disableAllHooks is true in this file, so no hooks run until you remove it")
if dry_run:
    print("dry run: nothing written")
    sys.exit(0)

directory = os.path.dirname(target)
os.makedirs(directory, exist_ok=True)
if original_text is not None:
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup, n = f"{target}.bak-hal-{stamp}", 1
    while os.path.exists(backup):  # never overwrite an earlier backup
        backup, n = f"{target}.bak-hal-{stamp}-{n}", n + 1
    shutil.copy2(target, backup)
    print(f"backup: {backup}")
fd, tmp = tempfile.mkstemp(prefix=".settings-", suffix=".json", dir=directory)
with os.fdopen(fd, "w", encoding="utf-8") as f:
    f.write(new_text)
if original_text is not None:
    shutil.copymode(target, tmp)
else:
    os.chmod(tmp, 0o600)
os.replace(tmp, target)
print("done. Restart Claude Code (or open /hooks) for the change to take effect.")
PY
