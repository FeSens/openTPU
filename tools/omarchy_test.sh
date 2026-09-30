#!/usr/bin/env bash
# Run pytest for the current worktree on the omarchy box instead of this machine.
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
# Ships the working tree (committed + uncommitted tracked files) to ~/otpu-test/<name> on omarchy,
# links the model checkpoints, and runs pytest there with ~/.local/bin/verilator and ~/otpu-venv.
# At most OTPU_REMOTE_JOBS (default 2) test runs share omarchy at once; a run waits for a free
# slot, so Vivado builds keep most of the CPU and memory (two builds plus three test runs thrashed 32 GB). The Verilator cache (build/) stays per tree on
# omarchy between runs of the same worktree. Exit status is pytest's.
set -euo pipefail
HOST=${OTPU_REMOTE:-omarchy.tail5bd214.ts.net}
JOBS=${OTPU_REMOTE_JOBS:-2}
EXTRA_KB=${OTPU_REMOTE_EXTRA_KB:-16000000}   # the extra slot's MemAvailable floor (16 GB)
mode=pytest
if [[ ${1:-} == --exec ]]; then mode=exec; shift; fi
top=$(git rev-parse --show-toplevel)
name=${OTPU_REMOTE_NAME:-$(basename "$top")-$(printf '%s' "$top" | shasum | cut -c1-6)}
dir="otpu-test/$name"
build=${OTPU_REMOTE_BUILD:-}
mark=${OTPU_REMOTE_SLOT_MARK:-}   # set: say on stderr when the run holds its slot (the tournament's
                                 # gate timeouts count from there)

# the tree as it is now: tracked files, including uncommitted edits
ssh "$HOST" "mkdir -p ~/$dir"
(cd "$top" && git ls-files -z | COPYFILE_DISABLE=1 tar --no-mac-metadata --no-xattrs --null -T - -cf -) | ssh "$HOST" "tar -xf - -C ~/$dir"

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
  (( \$(awk '/MemAvailable/{print \$2}' /proc/meminfo) > $EXTRA_KB )) && n=\$((n + 1))
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
