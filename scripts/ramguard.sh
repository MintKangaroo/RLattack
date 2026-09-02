#!/usr/bin/env bash
# Run a long RLAttack job under a hard memory cap.
#
# Training runs share this machine with other projects, and an unbounded run
# pushes the box into swap instead of failing fast. This wraps the command in a
# transient cgroup so the kernel kills the job -- and only the job -- when it
# exceeds its budget.
#
# Usage:  scripts/ramguard.sh [-m MAX] [-r RESERVE] -- <command> [args...]
#
#   -m MAX      hard memory cap for the job      (default $RLATTACK_MEM_MAX, 2G)
#   -r RESERVE  refuse to start unless this much
#               memory is available system-wide  (default $RLATTACK_MEM_FREE, 1G)
set -euo pipefail

max="${RLATTACK_MEM_MAX:-2G}"
reserve="${RLATTACK_MEM_FREE:-1G}"

while getopts ":m:r:" opt; do
  case "$opt" in
    m) max="$OPTARG" ;;
    r) reserve="$OPTARG" ;;
    *) echo "usage: $0 [-m MAX] [-r RESERVE] -- <command>" >&2; exit 2 ;;
  esac
done
shift $((OPTIND - 1))
[ "${1:-}" = "--" ] && shift
if [ "$#" -eq 0 ]; then
  echo "usage: $0 [-m MAX] [-r RESERVE] -- <command>" >&2
  exit 2
fi

# Accept 2G / 512M / 1500000000 and normalise to bytes for the availability check.
to_bytes() {
  local v="${1^^}"
  case "$v" in
    *K) echo $(( ${v%K} * 1024 )) ;;
    *M) echo $(( ${v%M} * 1024 * 1024 )) ;;
    *G) echo $(( ${v%G} * 1024 * 1024 * 1024 )) ;;
    *)  echo "$v" ;;
  esac
}

available=$(( $(awk '/^MemAvailable:/ {print $2}' /proc/meminfo) * 1024 ))
need=$(to_bytes "$reserve")
if [ "$available" -lt "$need" ]; then
  printf 'ramguard: only %s MiB available, need %s MiB free to start\n' \
    $(( available / 1048576 )) $(( need / 1048576 )) >&2
  exit 1
fi

if ! command -v systemd-run >/dev/null 2>&1; then
  echo "ramguard: systemd-run unavailable, running uncapped" >&2
  exec "$@"
fi

printf 'ramguard: cap=%s reserve=%s available=%s MiB\n' \
  "$max" "$reserve" $(( available / 1048576 )) >&2

# MemorySwapMax=0 keeps a runaway job from quietly eating swap instead of dying.
exec systemd-run --user --scope --quiet --collect \
  -p MemoryMax="$max" -p MemorySwapMax=0 -- "$@"
