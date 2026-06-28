#!/usr/bin/env bash
# Stop hook — advisory, loop-safe, read-only.
# Nudges to keep work_log.txt current: if tracked files changed this turn but
# work_log.txt was not touched, ask the model (once) to append a lean entry.
# Wired into .claude/settings.local.json under hooks.Stop.
set -u

# Loop guard: if this Stop is already a hook-driven continuation, do nothing.
# (Claude Code sets stop_hook_active=true on the continuation, so we nudge at
# most once per stop sequence and can never infinite-loop.)
input=$(cat)
case "$input" in
  *'"stop_hook_active":true'* | *'"stop_hook_active": true'*) exit 0 ;;
esac

# Read-only snapshot of the working tree (porcelain paths are repo-root-relative).
status=$(git status --porcelain 2>/dev/null) || exit 0
[ -n "$status" ] || exit 0

# Tracked changes (drop untracked "??" lines) excluding work_log.txt itself.
tracked=$(printf '%s\n' "$status" | grep -v '^??' | grep -v 'work_log\.txt')
[ -n "$tracked" ] || exit 0

# Already touched work_log.txt this turn? Stay silent.
printf '%s\n' "$status" | grep -q 'work_log\.txt' && exit 0

# Nudge exactly once. decision:block feeds the reason back to the model and lets
# it continue; the stop_hook_active guard above stops it from re-firing.
printf '%s' '{"decision":"block","reason":"You changed tracked files but did not touch work_log.txt. Per the repo habit, append a lean dated entry to work_log.txt summarizing what changed. If the change is trivial or still in progress, say so briefly and stop without logging."}'
exit 0
