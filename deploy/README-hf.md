---
title: RL Reward Shaping — 2-Link Arm
emoji: 🦾
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8501
pinned: false
license: mit
---

# Reward shaping on a custom 2-link arm

Showcase for [github.com/aghasalim/rl-arm-reward-shaping](https://github.com/aghasalim/rl-arm-reward-shaping).

A torque-controlled planar arm has to put its end effector within 5 cm of a
target, hold still there, and avoid an obstacle. Five reward functions; the agent
exploited two of them. The write-up leads with the exploits, not the final score.

To deploy: copy this file to `README.md` at the root of a Docker-SDK Space, push
the repo, and the Space builds from the `Dockerfile`.
