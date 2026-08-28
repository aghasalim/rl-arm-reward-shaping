"""Render the GIFs the showcase and README use.

Every clip uses the same three evaluation seeds, so "early vs late" and
"exploit vs fix" are genuinely like-for-like, a later checkpoint cannot look
better merely by drawing easier targets.
"""
from __future__ import annotations

from pathlib import Path

from stable_baselines3 import PPO

from .record import record

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"
FIG = ROOT / "reports" / "figures"

# (output name, model path or None for random, reward version)
CLIPS = [
    ("random", None, "v6_goalfocus"),
    ("v2_suicide", ART / "v2_distance_seed0.zip", "v2_distance"),
    ("v4_freeze", ART / "v4_potential_seed0.zip", "v4_potential"),
    ("final_early", ART / "final_v6_goalfocus_seed0_ckpt200000.zip", "v6_goalfocus"),
    ("final_mid", ART / "final_v6_goalfocus_seed0_ckpt2000000.zip", "v6_goalfocus"),
    ("final_late", ART / "final_v6_goalfocus_seed0.zip", "v6_goalfocus"),
]


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for name, path, version in CLIPS:
        if path is not None and not Path(path).exists():
            print(f"  skip {name}: {path} not found")
            continue
        model = PPO.load(path, device="cpu") if path else None
        record(model, FIG / f"{name}.gif", episodes=3, reward_version=version)


if __name__ == "__main__":
    main()
