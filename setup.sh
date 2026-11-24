#!/bin/bash
# One-time setup: create the project virtualenv, install Piper, fetch the
# voice model if it's missing, and run the tests. Safe to re-run.
#
#   ./setup.sh                 uses python3 from PATH (3.9 or newer)
#   PYTHON=/path/to/python3 ./setup.sh
set -euo pipefail

root="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python="${PYTHON:-python3}"

"$python" -c 'import sys; sys.exit(sys.version_info < (3, 9))' \
  || { echo "setup.sh: need Python 3.9 or newer (set PYTHON=...)" >&2; exit 1; }

if [ ! -x "$root/.venv/bin/python" ]; then
  echo "Creating $root/.venv with $("$python" --version)"
  "$python" -m venv "$root/.venv"
fi
"$root/.venv/bin/python" -m pip install --quiet --upgrade pip
"$root/.venv/bin/python" -m pip install --quiet -r "$root/requirements.txt"

"$root/scripts/fetch-model.sh"
mkdir -p "$root/state"

echo "Running unit tests..."
"$root/.venv/bin/python" -m unittest discover -s "$root/tests" -p 'test_textclean.py'

cat <<EOF

Ready. Try:  $root/bin/hal say "Good afternoon. Everything is running smoothly."
Then start Claude Code in $root, accept the trust prompt, and run /voice tap.
EOF
