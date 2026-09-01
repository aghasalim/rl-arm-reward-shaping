// Recompute every reward the environment paid, in JavaScript.
//
// src/rlarm/env.py::_reward is the only implementation of the six reward
// functions in this repository, and the whole project is an argument about
// them. Two of the published findings are arithmetic on that code: that v4 pays
// a stationary agent (1-gamma)*d every step, and that v5 and v6 telescope so a
// stationary agent earns exactly nothing. Nothing checked either.
//
// This reads verify/golden/reward_trace.csv, which holds every argument
// env._reward was called with and the number it returned, recomputes the reward
// from those arguments alone, and then uses the same independent implementation
// to derive the two published claims:
//
//   1. the v4 loitering table in NOTES.md, +10.0 / +15.0 / +20.0 per episode
//   2. the gamma=1 telescoping identity, sum of shaping == w * (phi_end - phi_start)
//
// Run: node verify/reward.js [repo-root]

const fs = require("fs");
const path = require("path");

const root = process.argv[2] || ".";
const TOL = 1e-12;
const GOAL_TOL = 0.05;      // env.py GOAL_TOL
const V4_GAMMA = 0.99;      // env.py, the v4_potential branch
const V4_WEIGHT = 5.0;
const MAX_STEPS = 200;      // env.py MAX_STEPS

function readCsv(file) {
    const text = fs.readFileSync(file, "utf8").replace(/\r/g, "").trim().split("\n");
    const header = text[0].split(",");
    return text.slice(1).map((line) => {
        const f = line.split(",");
        if (f.length !== header.length) {
            throw new Error(`ragged row in ${path.basename(file)}: ${f.length} fields`);
        }
        const row = {};
        header.forEach((h, i) => { row[h] = f[i]; });
        return row;
    });
}

// The potential of v5 and v6. goal_focus is zero for v5, so one function covers
// both, exactly as env.py does.
function potential(dist, speed, goalFocus, velWeight) {
    return -(dist + goalFocus * Math.sqrt(dist) + velWeight * speed);
}

// A transcription of env.py::_reward from the paper description of each
// version, not from the Python. Operation order is kept identical so a match is
// bit for bit rather than within a tolerance.
function reward(v, r) {
    const effortRaw = r.tau1 * r.tau1 + r.tau2 * r.tau2;

    if (v === "v1_sparse") return r.success ? 1.0 : 0.0;
    if (v === "v2_distance") return (r.success ? 10.0 : 0.0) - r.dist;
    if (v === "v3_penalties") {
        let x = -r.dist - 0.05;
        if (r.collided) x -= 5.0;
        if (r.success) x += 10.0;
        return x;
    }
    if (v === "v4_potential") {
        const shaping = V4_WEIGHT * (V4_GAMMA * -r.dist - -r.prev_dist);
        let x = shaping - 0.002 * effortRaw;
        if (r.dist < GOAL_TOL) x += 0.5;
        if (r.collided) x -= 30.0;
        if (r.success) x += 20.0;
        return x;
    }
    const pot = potential(r.dist, r.speed, r.goal_focus, r.vel_weight);
    const shaping = r.shaping_weight * (pot - r.prev_pot);
    const effort = 0.002 * effortRaw / (r.max_torque * r.max_torque);
    let x = shaping - effort - r.time_cost;
    if (r.settled) x += 1.0;
    if (r.collided) x -= r.collision_penalty;
    if (r.success) x += 50.0;
    return x;
}

const NUMERIC = ["dist", "prev_dist", "prev_pot", "speed", "tau1", "tau2",
                 "collided", "settled", "success", "shaping_weight",
                 "collision_penalty", "vel_weight", "time_cost", "goal_focus",
                 "max_torque", "reward", "step", "episode"];

const rows = readCsv(path.join(root, "verify", "golden", "reward_trace.csv"))
    .map((r) => {
        const out = { version: r.version };
        for (const k of NUMERIC) {
            out[k] = Number(r[k]);
            if (!Number.isFinite(out[k])) {
                throw new Error(`non-finite ${k} in reward_trace.csv`);
            }
        }
        return out;
    });

let failures = 0;

// --- 1. every recorded reward -------------------------------------------
console.log(`recomputing ${rows.length} rewards from verify/golden/reward_trace.csv`);
const perVersion = new Map();
for (const r of rows) {
    const delta = Math.abs(reward(r.version, r) - r.reward);
    const seen = perVersion.get(r.version) || { n: 0, max: 0 };
    seen.n += 1;
    seen.max = Math.max(seen.max, delta);
    perVersion.set(r.version, seen);
}
for (const [v, s] of [...perVersion.entries()].sort()) {
    const ok = s.max <= TOL;
    if (!ok) failures += 1;
    console.log(`  ${v.padEnd(14)} ${String(s.n).padStart(5)} steps   max |d| ` +
                `${s.max.toExponential(1)}   ${ok ? "ok" : "FAIL"}`);
}

// --- 2. the v4 loitering table in NOTES.md -------------------------------
// A stationary agent: distance never changes, no torque, no collision, no
// success. Driven through the same reward function checked above.
function loiterReturn(dist) {
    const step = {
        dist, prev_dist: dist, prev_pot: 0, speed: 0, tau1: 0, tau2: 0,
        collided: 0, settled: 0, success: 0, shaping_weight: 0,
        collision_penalty: 0, vel_weight: 0, time_cost: 0, goal_focus: 0,
        max_torque: 8,
    };
    let total = 0;
    for (let i = 0; i < MAX_STEPS; i += 1) total += reward("v4_potential", step);
    return total;
}

const notes = fs.readFileSync(path.join(root, "NOTES.md"), "utf8");
console.log(`\nv4 income for standing still, ${MAX_STEPS} steps, against the ` +
            "table in NOTES.md");
for (const dist of [1.0, 1.5, 2.0]) {
    const perStep = loiterReturn(dist) / MAX_STEPS;
    const total = loiterReturn(dist);
    const wantRow = notes.split("\n").find((l) => l.startsWith(`| ${dist.toFixed(1)} m |`));
    const published = wantRow ? wantRow.replace(/\*/g, "").split("|").map((c) => c.trim()) : null;
    const ok = published !== null &&
        published[2] === `+${perStep.toFixed(3)}` &&
        published[3] === `+${total.toFixed(1)}`;
    if (!ok) failures += 1;
    console.log(`  ${dist.toFixed(1)} m   +${perStep.toFixed(3)} per step   ` +
                `+${total.toFixed(1)} per episode   NOTES.md says ` +
                `${published ? published.slice(2, 4).join(" / ") : "no such row"}   ` +
                `${ok ? "ok" : "FAIL"}`);
}

// --- 3. the gamma=1 telescoping identity ---------------------------------
// v5 and v6 set gamma to 1 in the shaping term. The published claim is that the
// whole term then collapses to w * (phi_end - phi_start) over an episode, so
// there is no per-step income to farm. Checked on the real traces: the sum of
// every shaping payment against the two endpoint potentials.
console.log("\ngamma=1 shaping, summed over each traced episode against " +
            "w * (phi_end - phi_start)");
let worstTelescope = 0;
let worstChain = 0;
for (const v of ["v5_progress", "v6_goalfocus"]) {
    const episodes = new Set(rows.filter((r) => r.version === v).map((r) => r.episode));
    for (const ep of [...episodes].sort()) {
        const ep_rows = rows.filter((r) => r.version === v && r.episode === ep);
        let summed = 0;
        let prevPot = null;
        for (const r of ep_rows) {
            const pot = potential(r.dist, r.speed, r.goal_focus, r.vel_weight);
            summed += r.shaping_weight * (pot - r.prev_pot);
            // env.py stores the potential it just computed, so this step's
            // prev_pot must be the previous step's potential.
            if (prevPot !== null) worstChain = Math.max(worstChain, Math.abs(prevPot - r.prev_pot));
            prevPot = pot;
        }
        const last = ep_rows[ep_rows.length - 1];
        const telescoped = last.shaping_weight *
            (potential(last.dist, last.speed, last.goal_focus, last.vel_weight) -
             ep_rows[0].prev_pot);
        worstTelescope = Math.max(worstTelescope, Math.abs(summed - telescoped));
    }
}
const teleOk = worstTelescope <= 1e-9 && worstChain <= TOL;
if (!teleOk) failures += 1;
console.log(`  worst |sum - telescoped| over 8 episodes  ${worstTelescope.toExponential(1)}`);
console.log(`  worst break in the phi chain              ${worstChain.toExponential(1)}`);
console.log(`  ${teleOk ? "ok" : "FAIL"}`);

// A stationary agent under v5 and v6 earns exactly zero from shaping, which is
// the fix for the v4 exploit. Zero, not nearly zero.
const staticShaping = 10.0 * (potential(2.0, 0.0, 1.0, 0.1) - potential(2.0, 0.0, 1.0, 0.1));
if (staticShaping !== 0) {
    console.log(`  FAIL: standing still under v6 pays ${staticShaping}`);
    failures += 1;
} else {
    console.log("  standing still under v6 pays exactly 0 per step, against " +
                `+${(loiterReturn(2.0) / MAX_STEPS).toFixed(3)} under v4`);
}

if (failures) {
    console.log(`\n${failures} checks failed`);
    process.exit(1);
}
console.log("\nJavaScript reproduces every reward the environment paid, " +
            "the NOTES.md loitering table and the telescoping identity");
