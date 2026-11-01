#!/usr/bin/env bash
# Wait for pipeline_v0 to finish, then run flywheel iteration 0 -> 1 (+ control arm).
cd "$(dirname "$0")/.."
until grep -qE 'DONE|FAILED' outputs/logs/pipeline_v0.status 2>/dev/null; do sleep 20; done
grep -q DONE outputs/logs/pipeline_v0.status || exit 1
exec scripts/pipeline_flywheel.sh 0 20000
