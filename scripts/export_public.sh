#!/usr/bin/env bash
# Export the committed tree (HEAD) as a single, neutral-author commit and force-push it to the submission repo.
#
#   scripts/export_public.sh https://github.com/takerunakao/physical-ai-advanced1-tidy-flywheel.git
#
# `git archive HEAD` only takes tracked files, so gitignored course materials, data, outputs and .env never leave
# this machine. The export refuses to push if anything that looks like a name, e-mail or API key is in the tree.
set -euo pipefail
DEST=${1:?usage: export_public.sh <dest repo url>}
SRC=$(git rev-parse --show-toplevel)
if [ -n "$(git -C "$SRC" status --porcelain)" ]; then
  echo "commit or stash your changes first: the export takes HEAD, not the working tree" >&2; exit 1
fi
TMP=$(mktemp -d)
git -C "$SRC" archive --format=tar HEAD | tar -x -C "$TMP"
# Personal patterns (name, e-mail) come from the gitignored .personal_patterns, one regex per line, so this script
# carries none of them into the public tree; the key patterns are generic.
PAT="@gmail|/Users/[a-z]|sk-ant-|hf_[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_-]{30,}"
if [ -f "$SRC/.personal_patterns" ]; then PAT="$PAT|$(grep -v '^\s*$' "$SRC/.personal_patterns" | paste -sd '|' -)"; fi
if grep -rIn --exclude=export_public.sh --exclude=build_report.py -E "$PAT" "$TMP"; then
  echo "refusing to export: personal info or a key is in the tree (see above)" >&2; exit 1
fi
cd "$TMP"
git init -q -b main
git config user.name takerunakao
git config user.email 57853518+takerunakao@users.noreply.github.com
git add -A
git commit -q -m "Tidy flywheel: a VLM-supervised deployment loop for a tabletop tidying policy

Snapshot of the development repository for the Physical AI 応用編1 final assignment
(history squashed; source commit $(git -C "$SRC" rev-parse --short HEAD)).

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
git remote add origin "$DEST"
git push -f -q origin main
echo "pushed $(git rev-parse --short HEAD) ($(git ls-files | wc -l | tr -d ' ') files) to $DEST"
rm -rf "$TMP"
