#!/usr/bin/env bash
# Install OpenAI's tunnel-client (Secure MCP Tunnel) for linux amd64 into ~/.local, verified.
#
#   scripts/install-tunnel-client.sh [--force] [--provenance] [--version vX.Y.Z --sha256 HEX]
#
# - Downloads tunnel-client-<ver>-linux-amd64.zip and SHA256SUMS.txt from the GitHub release.
# - Refuses to install unless the zip's SHA-256 equals the hash pinned below (for the pinned version, or --sha256)
#   AND the entry in the release's SHA256SUMS.txt. A version without a pinned hash is checked against
#   SHA256SUMS.txt only, with a warning.
# - --provenance additionally runs `gh attestation verify` on the zip (needs gh >= 2.49 and network).
# - Installs the whole archive (tunnel-client expects cloudflared beside it) into
#   ~/.local/opt/tunnel-client/<ver>/ and points ~/.local/bin/tunnel-client at it (atomic symlink swap).
# - Idempotent: a verified install of the same version is left alone (no download) unless --force.
#
# Environment: TUNNEL_CLIENT_PREFIX (default ~/.local), TUNNEL_CLIENT_BASE_URL (default the GitHub release download
# URL; a file:// mirror works for offline tests; the hash checks still apply).
set -euo pipefail
umask 022

PINNED_VERSION="v0.0.15"                     # released 2026-09-25, tag commit a390c168ff1b2d14e73a95991c186c6aba3ff5a0
PINNED_SHA256="8c836dc5d68d68b663d9a5c5b28ff9fa780d9f7a3fffb1c306880b8f32fab5f1"   # tunnel-client-v0.0.15-linux-amd64.zip
PINNED_COMMIT="a390c168ff1b2d14e73a95991c186c6aba3ff5a0"
REPO="openai/tunnel-client"

version="$PINNED_VERSION"
expected=""
force=0
provenance=0

die() { echo "install-tunnel-client: $*" >&2; exit 1; }
say() { echo "install-tunnel-client: $*"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --version) [ $# -ge 2 ] || die "--version needs a value"; version="$2"; shift 2 ;;
    --sha256) [ $# -ge 2 ] || die "--sha256 needs a value"; expected="$2"; shift 2 ;;
    --force) force=1; shift ;;
    --provenance) provenance=1; shift ;;
    -h|--help) sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown argument: $1 (see --help)" ;;
  esac
done

[[ "$version" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "version must look like v1.2.3, got '$version'"
if [ -z "$expected" ] && [ "$version" = "$PINNED_VERSION" ]; then
  expected="$PINNED_SHA256"
fi
if [ -n "$expected" ]; then
  [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || die "--sha256 must be 64 lowercase hex characters"
fi

[ "$(uname -s)" = "Linux" ] || die "only Linux is supported here (macOS: brew install openai/tools/tunnel-client)"
case "$(uname -m)" in
  x86_64|amd64) ;;
  *) die "only linux amd64 is pinned here; for $(uname -m) use the release page or ghcr.io/openai/tunnel-client" ;;
esac
for tool in curl unzip sha256sum; do
  command -v "$tool" >/dev/null 2>&1 || die "missing required tool: $tool (sudo apt install $tool)"
done

if [ "$provenance" -eq 1 ]; then
  command -v gh >/dev/null 2>&1 || die "--provenance needs the GitHub CLI (gh >= 2.49)"
  gh attestation --help >/dev/null 2>&1 || die "--provenance needs gh >= 2.49 (this gh has no 'attestation' command)"
fi

prefix="${TUNNEL_CLIENT_PREFIX:-$HOME/.local}"
base_url="${TUNNEL_CLIENT_BASE_URL:-https://github.com/$REPO/releases/download/$version}"
asset="tunnel-client-$version-linux-amd64.zip"
opt_root="$prefix/opt/tunnel-client"
dest="$opt_root/$version"
bin_dir="$prefix/bin"
link="$bin_dir/tunnel-client"
marker="$dest/.verified-sha256"

reports_version() {  # binary: prints "0.0.15+<sha> ..." for v0.0.15
  local out
  out="$("$1" --version 2>/dev/null)" || return 1
  [[ "$out" == "${version#v}+"* ]]
}

installed_ok() {
  [ -f "$marker" ] && [ -x "$dest/tunnel-client" ] || return 1
  if [ -n "$expected" ]; then
    [ "$(cat "$marker")" = "$expected" ] || return 1
  fi
  reports_version "$dest/tunnel-client"
}

if [ "$force" -eq 0 ] && installed_ok; then
  if [ "$(readlink "$link" 2>/dev/null || true)" != "$dest/tunnel-client" ]; then
    mkdir -p "$bin_dir"
    ln -s "$dest/tunnel-client" "$link.tmp.$$" && mv -Tf "$link.tmp.$$" "$link"
    say "relinked $link -> $dest/tunnel-client"
  fi
  say "tunnel-client $version already installed and verified at $dest"
  exit 0
fi

if [ -n "${TUNNEL_CLIENT_BASE_URL:-}" ]; then
  say "WARNING: downloading from TUNNEL_CLIENT_BASE_URL=$base_url instead of the GitHub release"
fi

work="$(mktemp -d "${TMPDIR:-/tmp}/tunnel-client-install.XXXXXX")"
trap 'rm -rf "$work"' EXIT

say "downloading $asset and SHA256SUMS.txt ($version)"
curl --fail --location --silent --show-error --proto '=https,file' --proto-redir '=https' --tlsv1.2 -o "$work/$asset" "$base_url/$asset"
curl --fail --location --silent --show-error --proto '=https,file' --proto-redir '=https' --tlsv1.2 -o "$work/SHA256SUMS.txt" "$base_url/SHA256SUMS.txt"

actual="$(sha256sum "$work/$asset" | cut -d' ' -f1)"
listed="$(awk -v f="$asset" '$2 == f || $2 == "*" f { print $1 }' "$work/SHA256SUMS.txt")"
[ -n "$listed" ] || die "SHA256SUMS.txt has no entry for $asset: refusing to install"
[ "$(printf '%s\n' "$listed" | wc -l)" -eq 1 ] || die "SHA256SUMS.txt lists $asset more than once: refusing to install"
[ "$actual" = "$listed" ] || die "checksum MISMATCH for $asset: got $actual, SHA256SUMS.txt says $listed: refusing to install"
if [ -n "$expected" ]; then
  [ "$actual" = "$expected" ] || die "checksum MISMATCH for $asset: got $actual, pinned $expected: refusing to install"
  say "sha256 ok: $actual (matches the pinned hash and SHA256SUMS.txt)"
else
  say "WARNING: no pinned hash for $version; only the release's own SHA256SUMS.txt vouches for it ($actual)."
  say "         Pass --sha256 HEX from a trusted source, or use --provenance."
fi

if [ "$provenance" -eq 1 ]; then
  curl --fail --location --silent --show-error --proto '=https,file' --proto-redir '=https' --tlsv1.2 \
    -o "$work/provenance.sigstore.json" "$base_url/tunnel-client-$version-provenance.sigstore.json"
  commit_args=()
  if [ "$version" = "$PINNED_VERSION" ]; then
    commit_args=(--source-digest "$PINNED_COMMIT" --signer-digest "$PINNED_COMMIT")
  fi
  gh attestation verify "$work/$asset" --bundle "$work/provenance.sigstore.json" --repo "$REPO" \
    --signer-workflow "$REPO/.github/workflows/release.yml" --source-ref "refs/tags/$version" \
    --predicate-type https://slsa.dev/provenance/v1 --deny-self-hosted-runners "${commit_args[@]}" \
    || die "provenance verification FAILED: refusing to install"
  say "provenance ok (SLSA attestation from $REPO release workflow)"
fi

mkdir -p "$opt_root" "$bin_dir"
rm -rf "$opt_root"/*.partial.* "$opt_root"/*.old.*   # left by an interrupted earlier run
# Not a hidden mktemp dir: on the dev host `mktemp -d .../.name.XXXXXX` was SIGKILLed about half the time (endpoint
# security, CrowdStrike Falcon runs there), while a plain mkdir of a visible name never was.
stage="$opt_root/$version.partial.$$"
mkdir "$stage"
trap 'rm -rf "$work" "$stage"' EXIT
unzip -q "$work/$asset" -d "$stage"
[ -f "$stage/tunnel-client" ] || die "archive has no tunnel-client binary"
chmod 0755 "$stage/tunnel-client"
[ -f "$stage/cloudflared" ] && chmod 0755 "$stage/cloudflared"
reports_version "$stage/tunnel-client" || die "the unpacked binary does not report version $version"
printf '%s\n' "$actual" > "$stage/.verified-sha256"

if [ -e "$dest" ]; then
  rm -rf "$dest.old.$$"
  mv "$dest" "$dest.old.$$"
fi
mv "$stage" "$dest"
rm -rf "$dest.old.$$"
ln -s "$dest/tunnel-client" "$link.tmp.$$"
mv -Tf "$link.tmp.$$" "$link"

say "installed tunnel-client $version (sha256 $actual)"
say "  binary: $link -> $dest/tunnel-client"
case ":$PATH:" in
  *":$bin_dir:"*) ;;
  *) say "  note: $bin_dir is not on your PATH (the systemd unit uses the absolute path anyway)" ;;
esac
