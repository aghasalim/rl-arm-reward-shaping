//! Rerun the feasibility oracle from scratch and check the published table.
//!
//! The torque table in NOTES.md is the evidence for the central claim of this
//! project: that the arm was under-actuated and no reward function could have
//! fixed it. Ten numbers, all produced by one run of src/rlarm/oracle.py on top
//! of src/rlarm/env.py. If the dynamics, the inverse kinematics, the collision
//! test or the success criterion had a mistake in them, every one of those ten
//! numbers would be wrong in the same direction and nothing would notice.
//!
//! This is an independent implementation of all four, in Rust. The only thing
//! it takes from the Python is the task layouts, in
//! verify/golden/oracle_layouts.csv, because those come out of numpy's
//! generator. Everything after the layout is recomputed here: the manipulator
//! dynamics, RK4, forward kinematics, the segment to circle collision test, the
//! hold counter, the closed form IK and the PD law.
//!
//! It then does something the Python version cannot afford in CI. The published
//! rates are proportions over 200 fixed layouts, so they carry sampling error,
//! and nothing in the repository had measured it. A 100,000 draw bootstrap over
//! the episode outcomes puts an interval on the two headline numbers.
//!
//! Run: cd verify/oracle && cargo run --release -- <repo-root>

use std::env;
use std::f64::consts::PI;
use std::fs;
use std::process::exit;
use std::time::Instant;

// src/rlarm/env.py
const L1: f64 = 1.0;
const L2: f64 = 1.0;
const M1: f64 = 1.0;
const M2: f64 = 1.0;
const LC1: f64 = 0.5;
const LC2: f64 = 0.5;
const DAMPING: f64 = 0.10;
const MAX_VEL: f64 = 12.0;
const DT: f64 = 0.02;
const SUBSTEPS: usize = 2;
const MAX_STEPS: usize = 200;
const GOAL_TOL: f64 = 0.05;
const VEL_TOL: f64 = 0.10;
const HOLD_STEPS: usize = 10;
const OBSTACLE_R: f64 = 0.25;
// src/rlarm/oracle.py
const KP: f64 = 20.0;
const KD: f64 = 6.0;
const TORQUES: [f64; 5] = [2.0, 4.0, 6.0, 8.0, 12.0];

const BOOTSTRAP_DRAWS: usize = 100_000;

fn inertia(m: f64, l: f64) -> f64 {
    m * l * l / 12.0
}

fn mass_matrix(q2: f64) -> [[f64; 2]; 2] {
    let c2 = q2.cos();
    let m11 = inertia(M1, L1) + inertia(M2, L2) + M1 * LC1 * LC1
        + M2 * (L1 * L1 + LC2 * LC2 + 2.0 * L1 * LC2 * c2);
    let m12 = inertia(M2, L2) + M2 * (LC2 * LC2 + L1 * LC2 * c2);
    let m22 = inertia(M2, L2) + M2 * LC2 * LC2;
    [[m11, m12], [m12, m22]]
}

/// 2x2 solve with partial pivoting, matching what LAPACK does underneath
/// numpy.linalg.solve.
fn solve2(a: [[f64; 2]; 2], b: [f64; 2]) -> [f64; 2] {
    let (mut m, mut r) = (a, b);
    if m[1][0].abs() > m[0][0].abs() {
        m.swap(0, 1);
        r.swap(0, 1);
    }
    let f = m[1][0] / m[0][0];
    let u11 = m[1][1] - f * m[0][1];
    let x1 = (r[1] - f * r[0]) / u11;
    [(r[0] - m[0][1] * x1) / m[0][0], x1]
}

fn accel(s: [f64; 4], tau: [f64; 2]) -> [f64; 2] {
    let h = M2 * L1 * LC2 * s[1].sin();
    let cor = [-h * s[3] * s[3] - 2.0 * h * s[2] * s[3], h * s[2] * s[2]];
    solve2(
        mass_matrix(s[1]),
        [tau[0] - cor[0] - DAMPING * s[2], tau[1] - cor[1] - DAMPING * s[3]],
    )
}

fn deriv(s: [f64; 4], tau: [f64; 2]) -> [f64; 4] {
    let a = accel(s, tau);
    [s[2], s[3], a[0], a[1]]
}

fn wrap(a: f64) -> f64 {
    (a + PI).rem_euclid(2.0 * PI) - PI
}

fn integrate(s: &mut [f64; 4], tau: [f64; 2]) {
    let h = DT / SUBSTEPS as f64;
    for _ in 0..SUBSTEPS {
        let k1 = deriv(*s, tau);
        let mut t = [0.0; 4];
        for i in 0..4 {
            t[i] = s[i] + 0.5 * h * k1[i];
        }
        let k2 = deriv(t, tau);
        for i in 0..4 {
            t[i] = s[i] + 0.5 * h * k2[i];
        }
        let k3 = deriv(t, tau);
        for i in 0..4 {
            t[i] = s[i] + h * k3[i];
        }
        let k4 = deriv(t, tau);
        for i in 0..4 {
            s[i] += (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
    }
    s[2] = s[2].clamp(-MAX_VEL, MAX_VEL);
    s[3] = s[3].clamp(-MAX_VEL, MAX_VEL);
    s[0] = wrap(s[0]);
    s[1] = wrap(s[1]);
}

fn forward_kinematics(q1: f64, q2: f64) -> ([f64; 2], [f64; 2]) {
    let elbow = [L1 * q1.cos(), L1 * q1.sin()];
    let ee = [elbow[0] + L2 * (q1 + q2).cos(), elbow[1] + L2 * (q1 + q2).sin()];
    (elbow, ee)
}

fn seg_point_dist(a: [f64; 2], b: [f64; 2], p: [f64; 2]) -> f64 {
    let ab = [b[0] - a[0], b[1] - a[1]];
    let denom = ab[0] * ab[0] + ab[1] * ab[1];
    if denom < 1e-12 {
        return ((p[0] - a[0]).powi(2) + (p[1] - a[1]).powi(2)).sqrt();
    }
    let t = (((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / denom).clamp(0.0, 1.0);
    let c = [a[0] + t * ab[0], a[1] + t * ab[1]];
    ((p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2)).sqrt()
}

/// Closed form 2-link inverse kinematics, both elbow configurations.
fn inverse_kinematics(x: f64, y: f64, elbow_up: bool) -> (f64, f64) {
    let c2 = ((x * x + y * y - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)).clamp(-1.0, 1.0);
    let s2 = (1.0f64 - c2 * c2).max(0.0).sqrt() * if elbow_up { 1.0 } else { -1.0 };
    let q2 = s2.atan2(c2);
    let q1 = y.atan2(x) - (L2 * s2).atan2(L1 + L2 * c2);
    (q1, q2)
}

#[derive(Clone, Copy)]
struct Layout {
    target: [f64; 2],
    obstacle: [f64; 2],
    q: [f64; 2],
    /// What the Python oracle did on this layout, one (success, collision) pair
    /// per torque in TORQUES.
    python: [(bool, bool); TORQUES.len()],
}

/// One oracle episode. Returns (success, collision).
fn episode(l: Layout, use_obstacle: bool, max_torque: f64) -> (bool, bool) {
    let mut s = [l.q[0], l.q[1], 0.0, 0.0];

    let up = inverse_kinematics(l.target[0], l.target[1], true);
    let down = inverse_kinematics(l.target[0], l.target[1], false);
    let cost = |d: (f64, f64)| (d.0 - s[0]).abs_wrapped() + (d.1 - s[1]).abs_wrapped();
    // Python's min keeps the first argument on a tie, which is elbow up.
    let desired = if cost(down) < cost(up) { down } else { up };

    let collided_now = |s: &[f64; 4]| -> bool {
        if !use_obstacle {
            return false;
        }
        let (elbow, ee) = forward_kinematics(s[0], s[1]);
        seg_point_dist([0.0, 0.0], elbow, l.obstacle) < OBSTACLE_R
            || seg_point_dist(elbow, ee, l.obstacle) < OBSTACLE_R
    };

    let mut hold = 0usize;
    for step in 1..=MAX_STEPS {
        let err = [wrap(desired.0 - s[0]), wrap(desired.1 - s[1])];
        let act = [
            ((KP * err[0] - KD * s[2]) / max_torque).clamp(-1.0, 1.0),
            ((KP * err[1] - KD * s[3]) / max_torque).clamp(-1.0, 1.0),
        ];
        integrate(&mut s, [act[0] * max_torque, act[1] * max_torque]);

        let (_, ee) = forward_kinematics(s[0], s[1]);
        let dist = ((ee[0] - l.target[0]).powi(2) + (ee[1] - l.target[1]).powi(2)).sqrt();
        let collided = collided_now(&s);
        let settled = dist < GOAL_TOL && s[2].abs() < VEL_TOL && s[3].abs() < VEL_TOL;
        hold = if settled { hold + 1 } else { 0 };
        let success = hold >= HOLD_STEPS;

        if success || collided {
            return (success, collided);
        }
        if step >= MAX_STEPS {
            return (false, false);
        }
    }
    (false, false)
}

trait WrappedAbs {
    fn abs_wrapped(self) -> f64;
}
impl WrappedAbs for f64 {
    fn abs_wrapped(self) -> f64 {
        wrap(self).abs()
    }
}

/// xorshift64*. Not cryptographic and not meant to be: it needs to be uniform,
/// fast and reproducibly seeded so a failure can be re-run.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    fn below(&mut self, n: usize) -> usize {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        (x.wrapping_mul(0x2545_F491_4F6C_DD1D) % n as u64) as usize
    }
}

fn load_layouts(root: &str) -> Vec<(bool, Layout)> {
    let path = format!("{}/verify/golden/oracle_layouts.csv", root);
    let text = fs::read_to_string(&path).unwrap_or_else(|e| {
        eprintln!("cannot read {}: {}", path, e);
        exit(2);
    });
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header: Vec<&str> = lines.next().unwrap_or("").trim().split(',').collect();
    let col = |name: &str| {
        header.iter().position(|h| *h == name).unwrap_or_else(|| {
            eprintln!("oracle_layouts.csv has no column {}", name);
            exit(2);
        })
    };
    let (cu, ctx, cty, cox, coy, cq1, cq2) = (
        col("use_obstacle"), col("target_x"), col("target_y"),
        col("obstacle_x"), col("obstacle_y"), col("q1"), col("q2"),
    );

    let mut out = Vec::new();
    for line in lines {
        let f: Vec<&str> = line.trim().split(',').collect();
        if f.len() != header.len() {
            eprintln!("ragged row in oracle_layouts.csv");
            exit(2);
        }
        let num = |i: usize| -> f64 {
            f[i].trim().parse::<f64>().unwrap_or_else(|_| {
                eprintln!("field {} is not a number: {}", i, f[i]);
                exit(2);
            })
        };
        for i in 0..f.len() {
            if !num(i).is_finite() {
                eprintln!("non-finite field in oracle_layouts.csv: {}", f[i]);
                exit(2);
            }
        }
        let mut python = [(false, false); TORQUES.len()];
        for (k, t) in TORQUES.iter().enumerate() {
            let flag = |name: &str| -> bool {
                let v = num(col(name));
                if v != 0.0 && v != 1.0 {
                    eprintln!("{} is {}, which is not a flag", name, v);
                    exit(2);
                }
                v != 0.0
            };
            python[k] = (
                flag(&format!("success_t{}", *t as i64)),
                flag(&format!("collision_t{}", *t as i64)),
            );
        }
        out.push((
            num(cu) != 0.0,
            Layout {
                target: [num(ctx), num(cty)],
                obstacle: [num(cox), num(coy)],
                q: [num(cq1), num(cq2)],
                python,
            },
        ));
    }
    out
}

/// The rows of the MAX_TORQUE table in NOTES.md, as (torque, with, without)
/// strings exactly as published.
fn published_table(notes: &str) -> Vec<(f64, String, String)> {
    let mut rows = Vec::new();
    for line in notes.lines() {
        let plain = line.replace(['*', '`'], "");
        let cells: Vec<&str> = plain.split('|').map(|c| c.trim()).collect();
        if cells.len() != 5 || cells[0] != "" {
            continue;
        }
        let head: String = cells[1]
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '.')
            .collect();
        let torque: f64 = match head.parse() {
            Ok(t) => t,
            Err(_) => continue,
        };
        if cells[2].ends_with('%') && cells[3].ends_with('%') && TORQUES.contains(&torque) {
            rows.push((torque, cells[2].to_string(), cells[3].to_string()));
        }
    }
    rows
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let root = args.get(1).map(String::as_str).unwrap_or(".");

    let all = load_layouts(root);
    let with: Vec<Layout> = all.iter().filter(|(o, _)| *o).map(|(_, l)| *l).collect();
    let without: Vec<Layout> = all.iter().filter(|(o, _)| !*o).map(|(_, l)| *l).collect();
    if with.len() != 200 || without.len() != 200 {
        eprintln!("expected 200 layouts per setting, got {} and {}", with.len(), without.len());
        exit(2);
    }

    let notes = fs::read_to_string(format!("{}/NOTES.md", root)).unwrap_or_else(|e| {
        eprintln!("cannot read NOTES.md: {}", e);
        exit(2);
    });
    let published = published_table(&notes);
    if published.len() != TORQUES.len() {
        eprintln!(
            "found {} torque rows in NOTES.md, expected {}",
            published.len(),
            TORQUES.len()
        );
        exit(2);
    }

    let started = Instant::now();
    let mut failures = 0;
    let mut headline: Option<(Vec<bool>, Vec<bool>)> = None;

    println!("replaying the oracle over 200 layouts per configuration, in Rust");
    println!("  every episode outcome is compared with the Python oracle's, not just the rate");
    println!("  torque   success (obstacle)   success (no obstacle)   NOTES.md   \
episodes disagreeing");
    let mut disagreements = 0usize;
    let mut compared = 0usize;
    for (k, &t) in TORQUES.iter().enumerate() {
        let run = |ls: &[Layout], obs: bool| -> (Vec<bool>, Vec<bool>, usize) {
            let mut s = Vec::with_capacity(ls.len());
            let mut c = Vec::with_capacity(ls.len());
            let mut bad = 0usize;
            for l in ls {
                let (ok, hit) = episode(*l, obs, t);
                if (ok, hit) != l.python[k] {
                    bad += 1;
                }
                s.push(ok);
                c.push(hit);
            }
            (s, c, bad)
        };
        let (succ_o, coll_o, bad_o) = run(&with, true);
        let (succ_n, _, bad_n) = run(&without, false);
        compared += with.len() + without.len();
        disagreements += bad_o + bad_n;
        let rate = |v: &[bool]| v.iter().filter(|b| **b).count() as f64 / v.len() as f64;
        let (ro, rn) = (rate(&succ_o), rate(&succ_n));
        let (go, gn) = (format!("{:.1}%", 100.0 * ro), format!("{:.1}%", 100.0 * rn));

        let row = published.iter().find(|(pt, _, _)| *pt == t).unwrap();
        let ok = row.1 == go && row.2 == gn && bad_o + bad_n == 0;
        if !ok {
            failures += 1;
        }
        println!(
            "  {:>5.1}   {:>18}   {:>21}   {} / {}   {:>3}   {}",
            t, go, gn, row.1, row.2, bad_o + bad_n, if ok { "ok" } else { "FAIL" }
        );
        if t == 8.0 {
            headline = Some((succ_o, coll_o));
        }
    }
    println!("  {} episode outcomes compared, {} disagree", compared, disagreements);
    let elapsed = started.elapsed();

    // The two numbers the README quotes, from the same replay.
    let (succ, coll) = headline.unwrap();
    let srate = succ.iter().filter(|b| **b).count() as f64 / succ.len() as f64;
    let crate_ = coll.iter().filter(|b| **b).count() as f64 / coll.len() as f64;
    let readme = fs::read_to_string(format!("{}/README.md", root)).unwrap_or_default();
    println!("\nthe two numbers the README quotes for the oracle at torque 8.0");
    for (what, got, doc) in [
        ("success", format!("{:.1}%", 100.0 * srate), &readme),
        ("collision", format!("{:.1}%", 100.0 * crate_), &readme),
    ] {
        let ok = doc.contains(&got) && notes.contains(&got);
        if !ok {
            failures += 1;
        }
        println!(
            "  {:<10} {}   present in README.md and NOTES.md: {}",
            what, got, if ok { "yes" } else { "NO" }
        );
    }

    // How much of those two numbers is the luck of 200 layouts? Nothing in the
    // repository had asked. Resampling the episode outcomes is cheap here and
    // is why this runs on every push rather than once by hand.
    let mut rng = Rng::new(0x5EED_2026);
    let mut boot = |v: &[bool]| -> (f64, f64) {
        let mut stats = Vec::with_capacity(BOOTSTRAP_DRAWS);
        for _ in 0..BOOTSTRAP_DRAWS {
            let mut hits = 0usize;
            for _ in 0..v.len() {
                if v[rng.below(v.len())] {
                    hits += 1;
                }
            }
            stats.push(hits as f64 / v.len() as f64);
        }
        stats.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (stats[(0.025 * stats.len() as f64) as usize],
         stats[(0.975 * stats.len() as f64) as usize])
    };
    let (slo, shi) = boot(&succ);
    let (clo, chi) = boot(&coll);
    println!(
        "\n{} draw bootstrap over the 200 episode outcomes",
        BOOTSTRAP_DRAWS
    );
    println!("  success   {:.1}%  95% CI [{:.1}%, {:.1}%]", 100.0 * srate, 100.0 * slo, 100.0 * shi);
    println!("  collision {:.1}%  95% CI [{:.1}%, {:.1}%]", 100.0 * crate_, 100.0 * clo, 100.0 * chi);
    println!("  the whole 10 configuration table took {:?} here", elapsed);

    if failures > 0 {
        println!("\n{} rows disagree with the published table", failures);
        exit(1);
    }
    println!("\nRust reproduces all {} published oracle rates, and every one of the \
{} episode outcomes behind them, from the layouts alone",
             2 * TORQUES.len(), compared);
}
