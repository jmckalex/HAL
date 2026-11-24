#!/bin/bash
# Build the HAL menu-bar item: menubar/HALMenu.swift -> menubar/HALMenu.
# swiftc and AppKit only; no other dependencies. See AUTOSTART.md.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# Keep the compiler's module cache in the temp directory.
cache="${TMPDIR:-/tmp}/hal-menubar-module-cache"
swiftc -O -swift-version 6 -module-cache-path "$cache" -framework AppKit \
  -o "$here/HALMenu.new" "$here/HALMenu.swift"
# Replace, don't overwrite in place: a running copy keeps its own file.
mv -f "$here/HALMenu.new" "$here/HALMenu"
echo "built $here/HALMenu"
