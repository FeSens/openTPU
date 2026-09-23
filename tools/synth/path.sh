#!/bin/bash
# The critical path of a synthesized component: cells (every other line) and named signals.
# usage: path.sh <component>   (after tools/synth/run.py)
f="$(dirname "$0")/../../build/synth_board/$1.sta"
sed -n '/Latest arrival/,$p' "$f" | grep -E "^ +[0-9]+ " | awk '{print $1, $2, $3}' | awk 'NR%2==1' | head -24
sed -n '/Latest arrival/,$p' "$f" | grep -oE '\\[a-zA-Z_][a-zA-Z0-9_.]*' | awk '!seen[$0]++' | head -8
