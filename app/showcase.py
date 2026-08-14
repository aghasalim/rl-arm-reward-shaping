"""Streamlit showcase: the reward-shaping story, with the evidence attached."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIG = ROOT / "reports" / "figures"
ART = ROOT / "artifacts"

st.set_page_config(page_title="RL reward shaping — 2-link arm", layout="wide")


def show_gif(name: str, caption: str) -> None:
    p = FIG / f"{name}.gif"
    if p.exists():
        st.image(str(p), caption=caption, use_container_width=True)
    else:
        st.info(f"`{p.name}` not generated yet — run `make videos`.")


st.title("Teaching an arm to stop, not just arrive")
st.caption(
    "A 2-link torque-controlled arm must put its end effector within 5 cm of a "
    "target, hold still there for 10 steps, and not hit an obstacle — in 200 steps. "
    "Five reward functions, two of which the agent exploited."
)

results_path = ROOT / "reports" / "results.json"
results = json.loads(results_path.read_text()) if results_path.exists() else {}

if "final_multiseed" in results:
    ms = results["final_multiseed"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Success rate", f"{ms['mean']['success_rate']:.1%}",
              f"±{ms['std']['success_rate']:.1%} over {ms['n_seeds']} seeds")
    c2.metric("Collision rate", f"{ms['mean']['collision_rate']:.1%}")
    c3.metric("Final distance", f"{ms['mean']['mean_final_dist']:.3f} m")
    c4.metric("PD oracle reference", "52.0%", "hand-written controller")

tab1, tab2, tab3 = st.tabs(["The two exploits", "Training", "Reward versions"])

with tab1:
    st.subheader("Exploit 1 — deliberate self-termination")
    st.markdown(
        "`v2` paid `-distance` every step and never penalised collision. Crashing "
        "ends the episode, and ending the episode stops the bleeding. The agent "
        "found this in **40 out of 40** evaluation episodes, colliding at step ~75 "
        "where a random policy survives to ~110. `v3` added a `-5` collision "
        "penalty, which was still cheaper than surviving, so nothing changed."
    )
    show_gif("v2_suicide", "v2: the arm drives straight into the obstacle on purpose")

    st.subheader("Exploit 2 — farming the shaping drift")
    st.markdown(
        "`v4` used textbook potential-based shaping, `F = γΦ(s') − Φ(s)` with "
        "`Φ = −distance`. For a *stationary* agent this pays `(1−γ)·distance` every "
        "single step — **+20 per episode at 2 m, exactly the success bonus, at zero "
        "risk** — and it pays *more* the further away you loiter. The policy "
        "invariance theorem assumes an infinite horizon; under a 200-step cutoff "
        "the drift is just income. The agent parked at 1.4 m and collected it."
    )
    show_gif("v4_freeze", "v4: the arm drifts, then loiters — it is being paid to do this")

with tab2:
    st.subheader("Early vs late")
    a, b, c = st.columns(3)
    with a:
        show_gif("random", "Random policy — 0% success")
    with b:
        show_gif("final_early", "Early in training")
    with c:
        show_gif("final_late", "Final policy")
    for img, cap in [("reward_shaping_comparison.png", "Task success across reward versions"),
                     ("multiseed.png", "Per-seed spread on the final reward")]:
        p = FIG / img
        if p.exists():
            st.image(str(p), caption=cap, use_container_width=True)

with tab3:
    md = ROOT / "reports" / "results.md"
    if md.exists():
        st.markdown(md.read_text())
    else:
        st.info("Run `make eval` to generate reports/results.md.")

st.divider()
st.caption("Full decision trail in NOTES.md. Success criterion was fixed before "
           "any training and never adjusted to flatter a result.")
