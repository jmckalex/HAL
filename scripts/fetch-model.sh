#!/bin/bash
# Download the HAL 9000 Piper voice into models/ if it isn't there already.
# Source: https://huggingface.co/campwill/HAL-9000-Piper-TTS
#
#   scripts/fetch-model.sh           fetch missing files, verify checksums
#   scripts/fetch-model.sh --force   download again even if present
#
# The checksums are those of the files this project was set up and tested
# with. If the upstream files change, verification fails; set
# HAL_SKIP_CHECKSUM=1 to accept the new files anyway.
set -euo pipefail

root="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="$root/models"
base="https://huggingface.co/campwill/HAL-9000-Piper-TTS/resolve/main"
force=0
[ "${1:-}" = "--force" ] && force=1

expected_sha() {
  case "$1" in
    hal.onnx)      echo 0e08c82dc027bc72b8b839324801709e205e8c201ccae171b92e37a664d94361 ;;
    hal.onnx.json) echo cb6f82c03fc9e6db1f3b6978b9601bab9854f92c40d3a0681501944b8c26745e ;;
  esac
}

mkdir -p "$dest"
for name in hal.onnx hal.onnx.json; do
  file="$dest/$name"
  if [ -s "$file" ] && [ "$force" = 0 ]; then
    echo "$name: present"
  else
    echo "$name: downloading..."
    tmp="$(mktemp "$dest/.$name.XXXXXX")"
    trap 'rm -f "$tmp"' EXIT
    curl -fL --retry 3 --progress-bar -o "$tmp" "$base/$name"
    actual="$(shasum -a 256 "$tmp" | cut -d' ' -f1)"
    if [ "$actual" != "$(expected_sha "$name")" ] && [ "${HAL_SKIP_CHECKSUM:-0}" != 1 ]; then
      echo "$name: checksum mismatch (got $actual); upstream file may have changed." >&2
      echo "Re-run with HAL_SKIP_CHECKSUM=1 to accept it." >&2
      exit 1
    fi
    chmod 644 "$tmp"
    mv "$tmp" "$file"
    trap - EXIT
  fi
done

(cd "$dest" && for name in hal.onnx hal.onnx.json; do
  actual="$(shasum -a 256 "$name" | cut -d' ' -f1)"
  if [ "$actual" = "$(expected_sha "$name")" ]; then echo "$name: checksum OK"; else echo "$name: checksum differs from the tested file"; fi
done)
