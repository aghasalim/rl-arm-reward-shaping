.PHONY: setup test oracle shaping final eval plots videos showcase docker clean all

PY := .venv/bin/python
PIP := .venv/bin/pip

setup:
	python3.12 -m venv .venv
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -r requirements.txt pytest

test:
	$(PY) -m pytest tests/ -q

# Feasibility check: can a hand-written controller that already knows the answer
# satisfy the success criterion? Run this BEFORE blaming a reward function.
oracle:
	$(PY) -m src.rlarm.oracle

# The reward-shaping story: one seed per version, all in the same environment.
shaping:
	@for v in v1_sparse v2_distance v3_penalties v4_potential; do \
		OMP_NUM_THREADS=2 $(PY) -m src.rlarm.train --reward-version $$v --seed 0 \
			--timesteps 1200000 --checkpoints 100000 400000 & \
	done; wait

# Final policy: 5 seeds, reported with spread.
final:            ## the reported agent: v6 goal-focus, 3M steps, five seeds
	@for s in 0 1 2 3 4; do \
		OMP_NUM_THREADS=2 $(PY) -m src.rlarm.train --reward-version v6_goalfocus \
			--seed $$s --timesteps 3000000 --checkpoints 200000 2000000 \
			--tag final_v6_goalfocus & \
	done; wait

long8m:           ## the same agent run out to 8M, which is the comparison that got worse
	@for s in 0 1 2 3 4; do \
		OMP_NUM_THREADS=2 $(PY) -m src.rlarm.train --reward-version v6_goalfocus \
			--seed $$s --timesteps 8000000 --checkpoints 200000 2000000 3000000 \
			--tag long8m & \
	done; wait

eval:
	$(PY) -m src.rlarm.report

plots:
	$(PY) -m src.rlarm.plots

videos:
	$(PY) -m src.rlarm.make_videos

showcase:
	.venv/bin/streamlit run app/showcase.py

docker:
	docker build -t rl-arm-reward-shaping .

# Everything, in order. Hours of compute -- see the README before running.
all: test oracle shaping final eval plots videos

clean:
	rm -rf artifacts/*.zip artifacts/logs reports/figures reports/*.json reports/*.md
