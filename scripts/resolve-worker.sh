#!/usr/bin/env bash
# Lock evilflowers-worker to the ref this build's branch follows, for this CI run only (nothing is committed).
#
# The worker-kit contract of EvilFlowersCatalog/evilflowers-worker#13 (unified nightly + release tags), taken from
# evilflowers-analyzer-service's scripts/resolve-worker.sh until it ships as evilflowers-actions' `resolve-worker`
# action (EvilFlowersCatalog/evilflowers-actions#1):
#
#   develop, PRs, other branches → the kit's `master` head (dev follows master; the nightly rebuilds when it moves)
#   main / master, tags v*.*.*  → the kit's newest release tag v*.*.* (prod gets a kit commit that was released);
#                                  no tag found: `master` with a warning (rule 2 of evilflowers-worker#13)
#   KIT_REF=<ref> in the env     → that ref (a hold on a known-good tag, or a test of a kit branch); in CI the
#                                  repository variable KIT_REF, which also pauses the nightly
#
# Edits [tool.uv.sources] in pyproject.toml only when the ref is not master (branch = "master" → tag = "<tag>"), then
# runs `uv lock --upgrade-package evilflowers-worker`. The caller hands pyproject.toml + uv.lock to the image build.
#
# Output ($GITHUB_OUTPUT, or stdout): ref (master | v1.2.3), revision (the 40-hex commit uv.lock pins).
set -euo pipefail

cd "${PROJECT_DIR:-.}"
kit_url="${KIT_URL:-https://github.com/EvilFlowersCatalog/evilflowers-worker}"
out="${GITHUB_OUTPUT:-/dev/stdout}"

ref="${KIT_REF:-}"
if [ -z "$ref" ]; then
  case "${GITHUB_REF:-}" in
    refs/heads/main | refs/heads/master | refs/tags/*)
      # Newest semver tag; peeled (^{}) lines and pre-releases (v1.2.0-rc1) are not releases.
      ref=$(git ls-remote --tags --refs --sort=-version:refname "$kit_url" 'v*.*.*' \
        | cut -f2 | sed 's#^refs/tags/##' | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | head -n1 || true)
      if [ -z "$ref" ]; then
        echo "::warning::no release tag v*.*.* on evilflowers-worker; ${GITHUB_REF} builds on its master"
        ref=master
      fi
      ;;
    *) ref=master ;;
  esac
fi

if [ "$ref" != master ]; then
  grep -q 'evilflowers-worker = { git = .*, branch = "master" }' pyproject.toml \
    || { echo "::error::pyproject.toml: [tool.uv.sources] evilflowers-worker is not on branch = \"master\""; exit 1; }
  # A tag (v1.2.3) or another branch: uv's source keys are `tag` and `branch`.
  key=branch
  if git ls-remote --exit-code --tags "$kit_url" "refs/tags/$ref" > /dev/null; then key=tag; fi
  sed -i.bak -E "s#(evilflowers-worker = \{ git = \"[^\"]+\", )branch = \"master\"#\1${key} = \"${ref}\"#" pyproject.toml
  rm -f pyproject.toml.bak
fi

uv lock --upgrade-package evilflowers-worker
revision=$(grep -oE 'evilflowers-worker\?[^#"]*#[0-9a-f]{40}' uv.lock | head -n1 | cut -d'#' -f2)
[ -n "$revision" ] || { echo "::error::no evilflowers-worker commit in uv.lock"; exit 1; }

{
  echo "ref=$ref"
  echo "revision=$revision"
} >> "$out"
echo "::notice::evilflowers-worker ${ref} = ${revision}"
