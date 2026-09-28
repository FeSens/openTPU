#!/usr/bin/env bash
# Run pytest for the current worktree on the omarchy box instead of this machine.
#
#   tools/omarchy_test.sh [pytest args...]        e.g. tools/omarchy_test.sh -q tests/test_rtl.py -k fuzz
#
# Ships the working tree (committed + uncommitted tracked files) to ~/otpu-test/<name> on omarchy,
# links the model checkpoints, and runs pytest there with ~/.local/bin/verilator and ~/otpu-venv.
# At most OTPU_REMOTE_JOBS (default 3) test runs share omarchy at once; a run waits for a free
# slot, so Vivado builds keep most of the CPU. The Verilator cache (build/) stays per tree on
# omarchy between runs of the same worktree. Exit status is pytest's.
set -euo pipefail
HOST=${OTPU_REMOTE:-omarchy.tail5bd214.ts.net}
JOBS=${OTPU_REMOTE_JOBS:-3}
top=$(git rev-parse --show-toplevel)
name=$(basename "$top")-$(printf '%s' "$top" | shasum | cut -c1-6)
dir="otpu-test/$name"

# the tree as it is now: tracked files, including uncommitted edits
ssh "$HOST" "mkdir -p ~/$dir"
(cd "$top" && git ls-files -z | COPYFILE_DISABLE=1 tar --no-mac-metadata --null -T - -cf -) | ssh "$HOST" "tar -xf - -C ~/$dir"

args=$(printf '%q ' "$@")
ssh "$HOST" "bash -s" <<EOF
set -e
cd ~/$dir
ln -sfn ~/openTPU/models models
# one of JOBS slots (flock on per-slot lock files)
mkdir -p ~/otpu-test/.slots
for i in \$(seq 1 3600); do
  for s in \$(seq 1 $JOBS); do
    exec 9>~/otpu-test/.slots/\$s
    if flock -n 9; then
      export PATH=\$HOME/.local/bin:\$PATH
      nice -n 5 ~/otpu-venv/bin/python -m pytest -p no:cacheprovider $args
      exit \$?
    fi
  done
  sleep 2
done
echo "no free test slot on $HOST after 2 h" >&2; exit 2
EOF
