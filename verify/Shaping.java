// Is potential-based shaping actually policy invariant, and where does it stop?
//
// This project has two findings about Ng, Harada and Russell (1999). The first
// is that the theorem holds: F = gamma*Phi(s') - Phi(s) cannot change the
// optimal policy, which is why v4 was written that way. The second is that v4
// still got farmed, because the theorem assumes an infinite horizon with
// absorbing terminal states and the environment truncates at 200 steps.
//
// Both are stated in NOTES.md and neither was ever executed. This runs them.
// Value iteration on randomly generated MDPs, shaped and unshaped, and:
//
//   1. the shaped optimum must equal Q*(s,a) - Phi(s) exactly, and the greedy
//      policy must be identical, on every MDP
//   2. three deliberately broken variants of the shaping term must NOT be
//      invariant, or this check is passing on nothing
//   3. under a step limit, which values the cutoff at zero rather than at
//      -Phi, the shaping is exactly equivalent to paying Phi(s) at the moment
//      the clock runs out, and that changes which action is optimal
//
// Run: java verify/Shaping.java

import java.util.Random;

public class Shaping {

    static final int MDPS = 2000;
    static final int STATES = 8;      // the last one is terminal and absorbing
    static final int ACTIONS = 4;
    static final double GAMMA = 0.99; // env.py, the v4_potential branch
    static final int HORIZON = 200;   // env.py MAX_STEPS
    static final double VI_TOL = 1e-14;
    static final int VI_MAX = 200_000;
    static final double EXACT = 1e-9;

    // How the shaping term is built. POTENTIAL is the real one; the rest exist
    // so that a check which cannot fail is visibly a check which cannot fail.
    enum Form { POTENTIAL, TERMINAL_NOT_ZERO, GAMMA_DROPPED, NOT_A_POTENTIAL, NONE }

    static class Mdp {
        double[][][] p = new double[STATES][ACTIONS][STATES];
        double[][] r = new double[STATES][ACTIONS];
        double[] phi = new double[STATES];
        double[][] bonus = new double[STATES][ACTIONS];

        Mdp(Random rng) {
            for (int s = 0; s < STATES - 1; s++) {
                for (int a = 0; a < ACTIONS; a++) {
                    double total = 0;
                    for (int t = 0; t < STATES; t++) {
                        p[s][a][t] = rng.nextDouble();
                        total += p[s][a][t];
                    }
                    for (int t = 0; t < STATES; t++) {
                        p[s][a][t] /= total;
                    }
                    r[s][a] = 2 * rng.nextDouble() - 1;
                    bonus[s][a] = 2 * rng.nextDouble() - 1;
                }
            }
            // The terminal state absorbs and pays nothing.
            for (int a = 0; a < ACTIONS; a++) {
                p[STATES - 1][a][STATES - 1] = 1.0;
                r[STATES - 1][a] = 0.0;
            }
            for (int s = 0; s < STATES; s++) {
                phi[s] = 4 * rng.nextDouble() - 2;
            }
            phi[STATES - 1] = 0.0; // the assumption the theorem needs
        }

        double shaping(Form form, int s, int a, int t) {
            switch (form) {
                case NONE:
                    return 0.0;
                case POTENTIAL:
                    return GAMMA * phi[t] - phi[s];
                case TERMINAL_NOT_ZERO:
                    // Exactly the v4 failure: the terminal state is not
                    // absorbing, so its potential is never cancelled.
                    double pt = (t == STATES - 1) ? 1.5 : phi[t];
                    double ps = (s == STATES - 1) ? 1.5 : phi[s];
                    return GAMMA * pt - ps;
                case GAMMA_DROPPED:
                    return phi[t] - phi[s];
                default:
                    return bonus[s][a];
            }
        }
    }

    /** Value iteration. Returns Q, with V recoverable as the row maximum. */
    static double[][] solve(Mdp m, Form form) {
        double[] v = new double[STATES];
        double[][] q = new double[STATES][ACTIONS];
        for (int it = 0; it < VI_MAX; it++) {
            double delta = 0;
            for (int s = 0; s < STATES; s++) {
                double best = Double.NEGATIVE_INFINITY;
                for (int a = 0; a < ACTIONS; a++) {
                    double acc = m.r[s][a];
                    for (int t = 0; t < STATES; t++) {
                        if (m.p[s][a][t] != 0) {
                            acc += m.p[s][a][t] * (m.shaping(form, s, a, t) + GAMMA * v[t]);
                        }
                    }
                    q[s][a] = acc;
                    best = Math.max(best, acc);
                }
                // The terminal state is absorbing and worth nothing, in both
                // the shaped and the unshaped problem.
                if (s == STATES - 1) {
                    best = 0;
                }
                delta = Math.max(delta, Math.abs(best - v[s]));
                v[s] = best;
            }
            if (delta < VI_TOL) {
                break;
            }
        }
        return q;
    }

    /** One backward-induction sweep to a fixed horizon.
     *
     * cutoff is the value the episode is given when it runs out of steps. A
     * time limit is exactly this: the episode stops and whatever the agent was
     * standing in is worth zero. Returns the value function with `horizon`
     * steps still to go; policy[] is filled with the action chosen there.
     */
    static double[] finiteHorizon(Mdp m, Form form, int horizon, double[] cutoff,
                                  int[] policy) {
        double[] v = cutoff.clone();
        for (int step = 0; step < horizon; step++) {
            double[] next = new double[STATES];
            for (int s = 0; s < STATES; s++) {
                double best = Double.NEGATIVE_INFINITY;
                int arg = 0;
                for (int a = 0; a < ACTIONS; a++) {
                    double acc = m.r[s][a];
                    for (int t = 0; t < STATES; t++) {
                        acc += m.p[s][a][t] * (m.shaping(form, s, a, t) + GAMMA * v[t]);
                    }
                    if (acc > best) {
                        best = acc;
                        arg = a;
                    }
                }
                next[s] = best;
                if (policy != null) {
                    policy[s] = arg;
                }
            }
            v = next;
        }
        return v;
    }

    static int[] greedy(double[][] q) {
        int[] pi = new int[STATES];
        for (int s = 0; s < STATES; s++) {
            for (int a = 1; a < ACTIONS; a++) {
                if (q[s][a] > q[s][pi[s]]) {
                    pi[s] = a;
                }
            }
        }
        return pi;
    }

    static int differences(int[] a, int[] b) {
        int n = 0;
        for (int s = 0; s < STATES - 1; s++) {
            if (a[s] != b[s]) {
                n++;
            }
        }
        return n;
    }

    public static void main(String[] args) {
        Random rng = new Random(20260901L);
        Mdp[] mdps = new Mdp[MDPS];
        for (int i = 0; i < MDPS; i++) {
            mdps[i] = new Mdp(rng);
        }

        int failures = 0;
        System.out.printf("%d random MDPs, %d states, %d actions, gamma %.2f%n",
                MDPS, STATES, ACTIONS, GAMMA);

        // --- 1. the theorem -------------------------------------------
        double worstQ = 0;
        int changed = 0;
        for (Mdp m : mdps) {
            double[][] plain = solve(m, Form.NONE);
            double[][] shaped = solve(m, Form.POTENTIAL);
            for (int s = 0; s < STATES; s++) {
                for (int a = 0; a < ACTIONS; a++) {
                    worstQ = Math.max(worstQ, Math.abs(shaped[s][a] - (plain[s][a] - m.phi[s])));
                }
            }
            changed += differences(greedy(plain), greedy(shaped)) > 0 ? 1 : 0;
        }
        boolean ok = worstQ <= EXACT && changed == 0;
        failures += ok ? 0 : 1;
        System.out.println("\nF = gamma*Phi(s') - Phi(s), infinite horizon, "
                + "absorbing terminal with Phi = 0");
        System.out.printf("  worst |Q_shaped - (Q - Phi)|   %.2e%n", worstQ);
        System.out.printf("  MDPs whose optimal policy moved  %d of %d   %s%n",
                changed, MDPS, ok ? "ok" : "FAIL");

        // --- 2. the negative controls ---------------------------------
        System.out.println("\nvariants that are not potential-based, which must "
                + "move the policy or this proves nothing");
        for (Form form : new Form[] { Form.TERMINAL_NOT_ZERO, Form.GAMMA_DROPPED,
                                      Form.NOT_A_POTENTIAL }) {
            int moved = 0;
            for (Mdp m : mdps) {
                if (differences(greedy(solve(m, Form.NONE)), greedy(solve(m, form))) > 0) {
                    moved++;
                }
            }
            boolean broke = moved > 0;
            failures += broke ? 0 : 1;
            System.out.printf("  %-20s optimal policy moved on %4d of %d MDPs   %s%n",
                    form, moved, MDPS, broke ? "ok" : "FAIL, the check is vacuous");
        }

        // --- 3. what a step limit does to it --------------------------
        // A 200-step cutoff is a finite horizon whose value at the cutoff is
        // zero rather than -Phi, so the shaping no longer cancels. Working the
        // backward induction through by hand gives an exact statement of what
        // is left over:
        //
        //   V_shaped_H(s) + Phi(s) == V_plain_H(s) with the cutoff valued at Phi
        //
        // that is, shaping a truncated episode is the same thing as paying a
        // bonus of Phi(s) at the moment the clock runs out. Under v4's
        // Phi = -dist that bonus is worth the most where the arm never went, so
        // the sign of the whole effect is the exploit. Both halves are checked:
        // the identity has to hold to machine precision, and the shaped policy
        // has to actually differ from the unshaped one, or truncation would be
        // harmless and NOTES.md would be wrong.
        int[] horizons = { 1, 2, 5, 25, HORIZON };
        double worstIdentity = 0;
        int identityPolicyBreaks = 0;
        System.out.printf("%na cutoff valued at zero, which is what a %d-step "
                + "limit is%n", HORIZON);
        System.out.println("  horizon   policy moved   worst |V_shaped + Phi - "
                + "V_plain(cutoff = Phi)|");
        int movedTotal = 0;
        double[] zero = new double[STATES];
        for (int h : horizons) {
            int moved = 0;
            double worst = 0;
            for (Mdp m : mdps) {
                int[] piShaped = new int[STATES];
                int[] piPlain = new int[STATES];
                int[] piEquiv = new int[STATES];
                double[] vShaped = finiteHorizon(m, Form.POTENTIAL, h, zero, piShaped);
                finiteHorizon(m, Form.NONE, h, zero, piPlain);
                double[] vEquiv = finiteHorizon(m, Form.NONE, h, m.phi, piEquiv);
                for (int s = 0; s < STATES; s++) {
                    worst = Math.max(worst, Math.abs(vShaped[s] + m.phi[s] - vEquiv[s]));
                }
                if (differences(piShaped, piEquiv) > 0) {
                    identityPolicyBreaks++;
                }
                if (differences(piShaped, piPlain) > 0) {
                    moved++;
                }
            }
            worstIdentity = Math.max(worstIdentity, worst);
            movedTotal += moved;
            System.out.printf("  %7d   %4d of %d   %.2e%n", h, moved, MDPS, worst);
        }
        boolean identityOk = worstIdentity <= EXACT && identityPolicyBreaks == 0;
        failures += identityOk ? 0 : 1;
        System.out.printf("  the cutoff-bonus identity holds to %.2e on all %d "
                + "horizons, same policy every time: %s%n",
                worstIdentity, horizons.length, identityOk ? "ok" : "FAIL");
        boolean broke = movedTotal > 0;
        failures += broke ? 0 : 1;
        System.out.printf("  the shaped policy moved on %d MDP-horizons: %s%n", movedTotal,
                broke ? "ok, the invariance really does depend on the horizon"
                      : "FAIL, the horizon claim in NOTES.md does not hold");

        if (failures > 0) {
            System.out.printf("%n%d checks failed%n", failures);
            System.exit(1);
        }
        System.out.println("\nJava confirms the theorem exactly, and shows what a "
                + "step limit leaves behind");
    }
}
