#!/bin/bash
# Build the listener's audio helper: duplex/HALDuplex.swift -> duplex/hal-duplex.
# swiftc with AVFoundation and CoreAudio only; no other dependencies. See
# LISTENER.md, "Interrupting the voice".
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# Keep the compiler's module cache in the temp directory.
cache="${TMPDIR:-/tmp}/hal-duplex-module-cache"
swiftc -O -swift-version 5 -module-cache-path "$cache" \
  -framework AVFoundation -framework CoreAudio \
  -o "$here/hal-duplex.new" "$here/HALDuplex.swift"
# Replace, don't overwrite in place: a running copy keeps its own file.
mv -f "$here/hal-duplex.new" "$here/hal-duplex"
echo "built $here/hal-duplex"
