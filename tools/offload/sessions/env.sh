# Sourced by the card session scripts beside it (docs/offload.md 10, 11.5). Each path can be set
# from outside:
#   T           the openTPU checkout a session runs (default: the one holding this file; a
#               session's own tree is named in its header: git worktree add DIR SHA, then T=DIR)
#   O           the session directory: the checkpoints (or links to them), the pools, the HF
#               references (hf_reference.sh), the ISA simulator's (reference.sh) and the card's
#               configuration they were made with (REFCFG)
#   R           where the runs write: O/SESSION (a session script's SESSION, e.g. s8), or O
#   RF          the ISA simulator's references (default O): a tree's own, its programs' sums
#   OTPU_VENV   the Python environment (default ~/otpu-venv)
#   OTPU_QUIET  the host's quiet file (docs/host.md 8): no prebuild on the host while a session
#               runs; an outer script's claim stays
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
T=${T:-$(cd "$here/../../.." && pwd)}
O=${O:-$HOME/otpu-build/offload/card2}
R=${R:-$O${SESSION:+/$SESSION}}
RF=${RF:-$O}
REFCFG=${REFCFG:-$O/cfg-c2830d6a.pkl}
OTPU_QUIET=${OTPU_QUIET:-$HOME/otpu-build/QUIET}
export T O R RF REFCFG OTPU_QUIET PATH=${OTPU_VENV:-$HOME/otpu-venv}/bin:$PATH PYTHONPATH=$T
mkdir -p "$R"
q=$(cat "$OTPU_QUIET" 2>/dev/null)
[ -n "$q" ] && kill -0 "$q" 2>/dev/null || echo $$ > "$OTPU_QUIET"
trap '[ "$(cat "$OTPU_QUIET" 2>/dev/null)" = $$ ] && rm -f "$OTPU_QUIET"' EXIT
cd "$T" || exit 1
rev=$(cat COMMIT 2>/dev/null || git rev-parse --short HEAD)
mem() { awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo; }
