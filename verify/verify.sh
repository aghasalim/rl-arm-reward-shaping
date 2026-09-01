#!/usr/bin/env bash
# Recompute the numbers this repository publishes, in other languages, and
# require the answers to agree.
#
# Everything in the README, in NOTES.md and in reports/ came out of one
# simulator and one report script. The oracle table, the reward traces, the
# five-seed summary and the exploit arithmetic all read the same Python, so a
# mistake anywhere in it would be reproduced identically in every number
# downstream and nothing would disagree. These are separate implementations of
# the same arithmetic from the same inputs, and they have to land on the
# published values or this exits non-zero.
#
# Each check is skipped with a message if its toolchain is missing, so this
# runs on a laptop with only some of them installed. CI has all of them.
#
# Run: ./verify/verify.sh
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

pass=0 fail=0 skip=0
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

run () {
    local name="$1" tool="$2"; shift 2
    printf '\n=== %s ===\n' "$name"
    if ! command -v "$tool" >/dev/null 2>&1; then
        printf 'skipped: %s is not installed\n' "$tool"
        skip=$((skip + 1)); return
    fi
    if "$@"; then pass=$((pass + 1)); else fail=$((fail + 1)); fi
}

# --- Python -------------------------------------------------------------
# The fixtures the other seven read are dumped out of the environment itself.
# If env.py changes and they are not regenerated, every other check would keep
# agreeing with a stale copy of the simulator. Dump them again and require the
# committed files back: every integer exactly, every float to 1e-9 relative,
# which is the room LAPACK needs to give a different last ulp on another
# machine.
py="python3"
[ -x .venv/bin/python ] && py=".venv/bin/python"

have_python () { "$py" -c 'import gymnasium, numpy' >/dev/null 2>&1; }

check_python () {
    if ! have_python; then
        echo "skipped: $py cannot import gymnasium and numpy"
        return 2
    fi
    "$py" verify/export_golden.py --check
}

run_python () {
    printf '\n=== %s ===\n' "Python, golden fixtures are current"
    check_python
    case $? in
        0) pass=$((pass + 1)) ;;
        2) skip=$((skip + 1)) ;;
        *) fail=$((fail + 1)) ;;
    esac
}

# --- SQL ----------------------------------------------------------------
# sqlite3 reads stdin, which inside a script is the rest of this file, so the
# redirect from /dev/null is load bearing. Its CSV output is CRLF, hence the tr.
check_sql () {
    local out missing=0
    out="$(sqlite3 -init verify/summary.sql :memory: "" < /dev/null 2>/dev/null | tr -d '\r')"
    [ -n "$out" ] || { echo "sqlite3 produced nothing"; return 1; }
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        if ! grep -Fxq "$line" reports/results.md; then
            echo "not in reports/results.md: $line"
            missing=$((missing + 1))
        fi
    done <<< "$out"
    local n
    n=$(printf '%s\n' "$out" | grep -c '^|')
    if [ "$missing" -gt 0 ]; then
        echo "$missing of $n rebuilt rows are not in reports/results.md"
        return 1
    fi
    echo "SQL rebuilt all $n rows of reports/results.md from reports/results.json"
    return 0
}

check_c () {
    cc -std=c99 -O2 -Wall -Wextra -Wpedantic -Werror \
       -o "$tmp/physics" verify/physics.c -lm || return 1
    "$tmp/physics" "$root"
}

check_go   () { ( cd verify/gocheck && go run . -root "$root" ); }
check_rust () { ( cd verify/oracle && cargo run --release --quiet -- "$root" ); }
check_java () { ( cd "$tmp" && java "$root/verify/Shaping.java" ); }

run_python
run "SQL, the published tables"      sqlite3 check_sql
run "C, the dynamics kernel"         cc      check_c
run "Go, the results files"          go      check_go
run "R, the five-seed statistics"    Rscript Rscript verify/verify.R "$root"
run "Rust, the oracle replay"        cargo   check_rust
run "Java, shaping policy invariance" java   check_java
run "JavaScript, the reward functions" node  node verify/reward.js "$root"

printf '\n%s\n' "----------------------------------------"
printf '%d passed, %d failed, %d skipped\n' "$pass" "$fail" "$skip"
[ "$fail" -eq 0 ] || exit 1
[ "$pass" -gt 0 ] || { echo "nothing ran"; exit 1; }
