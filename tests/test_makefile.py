"""The README's reproduce path is `make shaping`, so it has to train every row
of the shaping table. report.py skips missing versions without a word."""
from pathlib import Path

from src.rlarm.env import REWARD_VERSIONS


def test_make_shaping_trains_every_reward_version():
    lines = (Path(__file__).parents[1] / "Makefile").read_text().splitlines()
    loop = next(line for line in lines[lines.index("shaping:"):] if "for v in" in line)
    assert set(loop.split(" in ", 1)[1].split(";")[0].split()) == set(REWARD_VERSIONS)
