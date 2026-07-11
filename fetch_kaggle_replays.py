"""Bulk-download Pokémon-TCG episode replays from Kaggle for BC/IL/RL.

Walks the simulation-competition chain the Kaggle CLI exposes:
    leaderboard -> team-submissions <teamId> -> episodes <submissionId> -> replay <episodeId>
and saves each replay as out/kaggle_replays/episode-<id>-replay.json (skipping any
already on disk). Kaggle regenerates episodes continuously, so re-running this daily
accretes fresh matches from the current top of the leaderboard.

Each replay is a kaggle_environments "cabt" episode: `rewards` (winner/loser),
`info.TeamNames`, and `steps[i][seat]` carrying the same fog `observation` our agents
consume plus the chosen `action` (option indices) and a full-board `visualize` dump.

Usage:
    python fetch_kaggle_replays.py                     # top 10 teams, 5 episodes each
    python fetch_kaggle_replays.py --teams 25 --episodes 10
    python fetch_kaggle_replays.py --team 16374395     # one specific team

Requires the Kaggle CLI (`kaggle`) authenticated (~/.kaggle/). No repo deps.
"""
import argparse
import csv
import io
import subprocess
import sys
from pathlib import Path

COMPETITION = "pokemon-tcg-ai-battle"


def kaggle_csv(args):
    """Run a `kaggle ... -v` command and return its rows as list[dict]."""
    proc = subprocess.run(
        ["kaggle", *args, "-v"], capture_output=True, text=True
    )
    if proc.returncode != 0:
        sys.exit(f"kaggle {' '.join(args)} failed:\n{proc.stderr or proc.stdout}")
    # The CSV output may be preceded by a "Next Page Token = ..." line.
    lines = [ln for ln in proc.stdout.splitlines() if "," in ln]
    if not lines:
        return []
    # Drop a leading page-token line if present (it has no header-like commas count match).
    if lines and lines[0].startswith("Next Page Token"):
        lines = lines[1:]
    return list(csv.DictReader(io.StringIO("\n".join(lines))))


def leaderboard_team_ids(n):
    rows = kaggle_csv(["competitions", "leaderboard", COMPETITION, "-s"])
    return [r["teamId"] for r in rows[:n]]


def best_submission_id(team_id):
    """Return the team's highest-public-score submission id (its live agent)."""
    rows = kaggle_csv(["competitions", "team-submissions", team_id])
    if not rows:
        return None
    best = max(rows, key=lambda r: float(r.get("publicScore") or "nan"))
    return best["id"]


def episode_ids(submission_id, limit):
    rows = kaggle_csv(["competitions", "episodes", submission_id])
    done = [r for r in rows if "COMPLETED" in (r.get("state") or "")]
    return [r["id"] for r in done[:limit]]


def download_replay(episode_id, out_dir):
    dest = out_dir / f"episode-{episode_id}-replay.json"
    if dest.exists():
        return False  # already have it
    proc = subprocess.run(
        ["kaggle", "competitions", "replay", episode_id, "-p", str(out_dir), "-q"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not dest.exists():
        print(f"  ! replay {episode_id} failed: {proc.stderr.strip() or proc.stdout.strip()}")
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teams", type=int, default=10, help="how many top-leaderboard teams to pull (default 10)")
    ap.add_argument("--team", type=str, default=None, help="pull a single team id instead of the leaderboard")
    ap.add_argument("--episodes", type=int, default=5, help="most-recent episodes per submission (default 5)")
    ap.add_argument("--out", type=Path, default=Path("out/kaggle_replays"), help="download dir")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    team_ids = [args.team] if args.team else leaderboard_team_ids(args.teams)
    print(f"{len(team_ids)} team(s); up to {args.episodes} episodes each -> {args.out}")

    got = 0
    for team_id in team_ids:
        sub = best_submission_id(team_id)
        if not sub:
            print(f"team {team_id}: no submissions")
            continue
        eps = episode_ids(sub, args.episodes)
        new = sum(download_replay(e, args.out) for e in eps)
        got += new
        print(f"team {team_id} sub {sub}: {len(eps)} episodes, {new} new")

    total = len(list(args.out.glob("episode-*-replay.json")))
    print(f"done: {got} new replays; {total} total in {args.out}")


if __name__ == "__main__":
    main()
