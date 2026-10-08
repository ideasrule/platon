#!/bin/bash
set -euo pipefail

# Run from the repository root, even when invoked from elsewhere.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

# Requires: python3 -m pip install build twine
# Only upload artifacts produced by this invocation.
release_dir=$(mktemp -d "${TMPDIR:-/tmp}/platon-release.XXXXXX")
trap 'rm -rf -- "$release_dir"' EXIT
python3 -m build --outdir "$release_dir"
python3 -m twine check "$release_dir"/*
python3 -m twine upload "$release_dir"/*
