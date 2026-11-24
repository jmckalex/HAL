#!/bin/bash
# One-time setup for the always-listening front end (bin/hal-listen):
#   - a separate virtualenv, .venv-listen (the voice's .venv is not touched)
#   - the Silero VAD model, models/silero_vad.onnx (2.3 MB, MIT)
#   - the Parakeet speech-to-text model, models/parakeet-tdt-0.6b-v2/ (2.5 GB, CC-BY-4.0)
#   - the listener's unit tests
# Safe to re-run. Nothing is written outside the project: pip runs without its
# cache, and model downloads go to models/ (HuggingFace's cache to models/hf/).
# If ~/.cache/huggingface already holds the Parakeet model it is cloned from
# there (an APFS copy-on-write clone: instant, no extra disk space).
#
#   scripts/setup-listen.sh                  uses python3 from PATH (3.10 or newer)
#   PYTHON=/path/to/python3 scripts/setup-listen.sh
set -euo pipefail

root="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="${PYTHON:-python3}"
venv="$root/.venv-listen"
models="$root/models"
export PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
export HF_HOME="$models/hf" HF_HUB_DISABLE_TELEMETRY=1

[ "$(uname -s)-$(uname -m)" = "Darwin-arm64" ] \
  || { echo "setup-listen.sh: Parakeet runs on MLX, which needs Apple silicon" >&2; exit 1; }
"$python" -c 'import sys; sys.exit(sys.version_info < (3, 10))' \
  || { echo "setup-listen.sh: need Python 3.10 or newer (set PYTHON=...)" >&2; exit 1; }

if [ ! -x "$venv/bin/python" ]; then
  echo "Creating $venv with $("$python" --version)"
  "$python" -m venv "$venv"
fi
"$venv/bin/python" -m pip install --quiet --upgrade pip
"$venv/bin/python" -m pip install --quiet -r "$root/requirements-listen.txt"

sha256() { shasum -a 256 "$1" | cut -d' ' -f1; }

# Silero VAD: taken from the silero-vad wheel on PyPI (the package itself needs torch; we don't).
vad="$models/silero_vad.onnx"
vad_sha=1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3
if [ ! -s "$vad" ] || [ "$(sha256 "$vad")" != "$vad_sha" ]; then
  echo "Fetching the Silero VAD model"
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  "$venv/bin/python" -m pip download --quiet --no-deps silero-vad==6.2.0 -d "$tmp"
  (cd "$tmp" && unzip -q -o silero_vad-6.2.0-py3-none-any.whl silero_vad/data/silero_vad.onnx)
  [ "$(sha256 "$tmp/silero_vad/data/silero_vad.onnx")" = "$vad_sha" ] \
    || { echo "setup-listen.sh: silero_vad.onnx checksum mismatch" >&2; exit 1; }
  mkdir -p "$models"
  mv "$tmp/silero_vad/data/silero_vad.onnx" "$vad"
fi
echo "silero_vad.onnx: present"

# Parakeet TDT 0.6B v2 (MLX conversion).
asr="$models/parakeet-tdt-0.6b-v2"
asr_sha=b958c37a6baa6874a279108755c8f2818e27bf647d72d54800a234a421341dfe
if [ ! -s "$asr/model.safetensors" ] || [ ! -s "$asr/config.json" ]; then
  mkdir -p "$asr"
  cached="$(ls -d "$HOME"/.cache/huggingface/hub/models--mlx-community--parakeet-tdt-0.6b-v2/snapshots/*/ 2>/dev/null | head -1 || true)"
  if [ -n "$cached" ] && [ -s "$cached/model.safetensors" ]; then
    echo "Cloning the Parakeet model from $cached"
    cp -c -L "$cached/config.json" "$cached/model.safetensors" "$asr/" 2>/dev/null \
      || cp -L "$cached/config.json" "$cached/model.safetensors" "$asr/"
  else
    echo "Downloading the Parakeet model (2.5 GB)"
    "$venv/bin/python" -c 'import sys; from huggingface_hub import snapshot_download
snapshot_download("mlx-community/parakeet-tdt-0.6b-v2", local_dir=sys.argv[1], allow_patterns=["config.json", "model.safetensors"])' "$asr"
  fi
fi
if [ "$(sha256 "$asr/model.safetensors")" != "$asr_sha" ]; then
  echo "setup-listen.sh: warning: model.safetensors differs from the tested file" >&2
fi
echo "parakeet-tdt-0.6b-v2: present"

mkdir -p "$root/state"
echo "Running the listener's unit tests..."
"$root/.venv/bin/python" -m unittest discover -s "$root/tests" -p 'test_listen*.py' 2>&1 | tail -3 \
  || "$venv/bin/python" -m unittest discover -s "$root/tests" -p 'test_listen*.py' 2>&1 | tail -3

cat <<EOF

Ready. Check the microphone (levels only; nothing is kept):  $root/bin/hal-listen mic-check
Try it without a session:  $root/bin/hal-listen run --dry-run --silent
See LISTENER.md for wiring it into \`hal claude\`.
EOF
