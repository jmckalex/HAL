#!/bin/bash
# Fetch the speaker-embedding model for the voice check (hal/listener/voiceprint.py):
# WeSpeaker ResNet34-LM trained on VoxCeleb, as converted to ONNX by sherpa-onnx
# (Apache-2.0), 26.5 MB, into models/speaker/. Then `hal-listen enrol`.
#
#   scripts/fetch-voice-model.sh           download it if it isn't there
#   scripts/fetch-voice-model.sh --force   download it again
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
dir="$root/models/speaker"
name="wespeaker_en_voxceleb_resnet34_LM.onnx"
url="https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/$name"
sha="e9848563da86f263117134dfd7ad63c92355b37de492b55e325400c9d9c39012"

if [[ -f "$dir/$name" && "${1:-}" != "--force" ]]; then
    echo "$name: present"
    exit 0
fi
mkdir -p "$dir"
tmp="$(mktemp "$dir/.download.XXXXXX")"
trap 'rm -f "$tmp"' EXIT
echo "$name: downloading..."
curl -fL --retry 3 --progress-bar -o "$tmp" "$url"
got="$(shasum -a 256 "$tmp" | cut -d' ' -f1)"
if [[ "$got" != "$sha" ]]; then
    echo "$name: checksum mismatch ($got); not installed" >&2
    exit 1
fi
mv "$tmp" "$dir/$name"
trap - EXIT
echo "$name: installed in models/speaker/"
