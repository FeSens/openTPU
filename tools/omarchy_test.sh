#!/usr/bin/env bash
# Run pytest for this script's worktree on the omarchy box instead of this machine.
#
#   tools/omarchy_test.sh [pytest args...]        e.g. tools/omarchy_test.sh -q tests/test_rtl.py -k fuzz
#   tools/omarchy_test.sh --exec CMD [args...]    any command in the tree instead of pytest (with
#                                                 the venv and Verilator on PATH), e.g.
#                                                 --exec env OTPU_AXI=1 python -m pytest -q tests/test_board.py
#
# Optional: OTPU_REMOTE_NAME=<name> names the tree (~/otpu-test/<name>, default: the worktree's
# directory name + a hash of its path); OTPU_REMOTE_BUILD=<dir under ~> links the tree's build/
# (the Verilator cache, keyed by source hash) to a shared directory, so fresh trees reuse it.
#
# Ships the worktree the script is in (not the current directory's: `bash /path/to/tree/tools/
# omarchy_test.sh` tests that tree), its tracked files with their uncommitted edits, to
# ~/otpu-test/<name> on omarchy and says which tree and commit on stderr; files there that the
# worktree no longer tracks are deleted (build/, models and __pycache__ stay), so a module
# removed here is gone there too. Links the model checkpoints and runs pytest there with
# ~/.local/bin/verilator and ~/otpu-venv.
# At most OTPU_REMOTE_JOBS (default 2) test runs share omarchy at once; a run waits for a free
# slot, so Vivado builds keep most of the CPU and memory (two builds plus three test runs
# thrashed 32 GB). The Verilator cache (build/) stays per tree on omarchy between runs of the
# same worktree. Exit status is pytest's.
set -euo pipefail
HOST=${OTPU_REMOTE:-omarchy.tail5bd214.ts.net}
JOBS=${OTPU_REMOTE_JOBS:-2}
EXTRA_KB=${OTPU_REMOTE_EXTRA_KB:-16000000}   # the extra slot's MemAvailable floor (16 GB)
mode=pytest
if [[ ${1:-} == --exec ]]; then mode=exec; shift; fi
top=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && git rev-parse --show-toplevel)
name=${OTPU_REMOTE_NAME:-$(basename "$top")-$(printf '%s' "$top" | shasum | cut -c1-6)}
dir="otpu-test/$name"
build=${OTPU_REMOTE_BUILD:-}
mark=${OTPU_REMOTE_SLOT_MARK:-}   # set: say on stderr when the run holds its slot (the tournament's
                                 # gate timeouts count from there)
edits=$(git -C "$top" status --porcelain --untracked-files=no | wc -l | tr -d ' ')
echo "omarchy_test: $top at $(git -C "$top" rev-parse --short HEAD)$([[ $edits == 0 ]] ||
  echo " + $edits uncommitted") -> $HOST:~/$dir" >&2

# the tree as it is now: tracked files, including uncommitted edits; then the files it no longer
# tracks go (the list inline, then the pruning script)
ssh "$HOST" "mkdir -p ~/$dir"
(cd "$top" && git ls-files -z | COPYFILE_DISABLE=1 tar --no-mac-metadata --no-xattrs --null -T - -cf -) | ssh "$HOST" "tar -xf - -C ~/$dir"
{ printf 'cd ~/%s\ncat > .otpu-files <<"OTPU_FILES_END"\n' "$dir"
  git -C "$top" -c core.quotePath=false ls-files
  printf 'OTPU_FILES_END\n'; cat <<'PRUNE'
LC_ALL=C sort -o .otpu-files .otpu-files
find . \( -path ./build -o -path ./models -o -path ./obj_dir -o -name __pycache__ \
  -o -name .pytest_cache \) -prune -o \( -type f -o -type l \) -print | sed 's|^\./||' \
  | grep -vx .otpu-files | LC_ALL=C sort | LC_ALL=C comm -23 - .otpu-files \
  | while IFS= read -r f; do rm -f -- "$f"; done
# a package whose sources all went: its __pycache__ too (else it imports as a namespace
# package), then the empty directories
find . \( -path ./build -o -path ./obj_dir \) -prune -o -name __pycache__ -type d -print \
  | while IFS= read -r d; do [ -n "$(ls -A "$(dirname "$d")" | grep -vx __pycache__)" ] || rm -rf "$d"; done
find . -mindepth 1 \( -path ./build -o -path ./obj_dir \) -prune -o -type d -empty -print \
  | while IFS= read -r d; do rmdir -p "$d" 2> /dev/null || true; done
PRUNE
} | ssh "$HOST" "bash -s"

args=$(printf '%q ' "$@")
ssh "$HOST" "bash -s" <<EOF
set -e
cd ~/$dir
ln -sfn ~/openTPU/models models
if [[ -n "$build" ]]; then mkdir -p ~/$build; rm -rf build; ln -sfn ~/$build build; fi
# one of JOBS slots (flock on per-slot lock files), plus one more while omarchy has memory to spare
# (MemAvailable above EXTRA_KB, i.e. no second full build placing)
mkdir -p ~/otpu-test/.slots
for i in \$(seq 1 3600); do
  n=$JOBS
  (( \$(awk '/MemAvailable/{print \$2}' /proc/meminfo 2> /dev/null || echo 0) > $EXTRA_KB )) && n=\$((n + 1))
  for s in \$(seq 1 \$n); do
    exec 9>~/otpu-test/.slots/\$s
    if flock -n 9; then
      if [[ -n "$mark" ]]; then echo "omarchy_test: slot \$s" >&2; fi
      export PATH=\$HOME/.local/bin:\$PATH
      if [[ $mode == exec ]]; then
        export PATH=\$HOME/otpu-venv/bin:\$PATH PYTHONPATH=\$PWD
        nice -n 5 $args
      else
        nice -n 5 ~/otpu-venv/bin/python -m pytest -p no:cacheprovider $args
      fi
      exit \$?
    fi
  done
  sleep 2
done
echo "no free test slot on $HOST after 2 h" >&2; exit 2
EOF
