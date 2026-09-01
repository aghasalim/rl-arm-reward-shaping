/* Reintegrate the arm, in C, and check it lands where the Python said it did.
 *
 * Every number in this repository is measured inside one simulator. If the
 * manipulator dynamics or the RK4 step in src/rlarm/env.py were wrong, the
 * success rates, the oracle table and the reward traces would all be wrong
 * together and nothing would disagree, because there is only one of it.
 *
 * This is a second implementation of the same equations (Spong ch. 7, the
 * closed form two link mass matrix) with its own 2x2 solve and its own RK4, and
 * it is required to reproduce verify/golden/physics_trace.csv.
 *
 * It also measures the claim the README makes about the integrator: that
 * semi-implicit Euler at dt=0.02 injects energy into the unforced arm and RK4
 * does not. tests/test_physics.py asserts the RK4 half. The Euler half was
 * stated and never measured here.
 *
 * Build: cc -std=c99 -O2 -o physics verify/physics.c -lm
 * Run:   ./physics <repo-root>
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LINE 4096
#define MAX_ROWS 512
#define MAX_CASES 16

/* Geometry and integration constants, from src/rlarm/env.py. */
static const double L1 = 1.0, L2 = 1.0;
static const double M1 = 1.0, M2 = 1.0;
static const double LC1 = 0.5, LC2 = 0.5;
static const double DT = 0.02;
static const int SUBSTEPS = 2;
static const double MAX_VEL = 12.0;
static const double TOL = 1e-9;
/* glibc only exposes PI outside strict C99, and this is built with
 * -std=c99 -Wpedantic -Werror, so carry the constant rather than the guard. */
static const double PI = 3.14159265358979323846;

static double inertia(double m, double l) { return m * l * l / 12.0; }

static void mass_matrix(double q2, double m[2][2])
{
    const double c2 = cos(q2);
    m[0][0] = inertia(M1, L1) + inertia(M2, L2) + M1 * LC1 * LC1
            + M2 * (L1 * L1 + LC2 * LC2 + 2.0 * L1 * LC2 * c2);
    m[0][1] = inertia(M2, L2) + M2 * (LC2 * LC2 + L1 * LC2 * c2);
    m[1][0] = m[0][1];
    m[1][1] = inertia(M2, L2) + M2 * LC2 * LC2;
}

/* 2x2 solve with partial pivoting, which is what LAPACK does underneath
 * numpy.linalg.solve. Cramer's rule would also be correct here and would drift
 * from the Python by a few ulp per step for no reason. */
/* a is not const: passing a plain double[2][2] to a const double[2][2]
 * parameter is a constraint violation in C99, which gcc -Wpedantic
 * rejects, unlike the same thing one level down for double[]. */
static void solve2(double a[2][2], const double b[2], double x[2])
{
    double m[2][2] = {{a[0][0], a[0][1]}, {a[1][0], a[1][1]}};
    double r[2] = {b[0], b[1]};
    if (fabs(m[1][0]) > fabs(m[0][0])) {
        double t;
        t = m[0][0]; m[0][0] = m[1][0]; m[1][0] = t;
        t = m[0][1]; m[0][1] = m[1][1]; m[1][1] = t;
        t = r[0];    r[0] = r[1];       r[1] = t;
    }
    const double f = m[1][0] / m[0][0];
    const double u11 = m[1][1] - f * m[0][1];
    const double y1 = r[1] - f * r[0];
    x[1] = y1 / u11;
    x[0] = (r[0] - m[0][1] * x[1]) / m[0][0];
}

static void accel(const double s[4], const double tau[2], double damping, double out[2])
{
    double m[2][2];
    mass_matrix(s[1], m);
    const double h = M2 * L1 * LC2 * sin(s[1]);
    const double cor[2] = { -h * s[3] * s[3] - 2.0 * h * s[2] * s[3], h * s[2] * s[2] };
    const double rhs[2] = { tau[0] - cor[0] - damping * s[2],
                            tau[1] - cor[1] - damping * s[3] };
    solve2(m, rhs, out);
}

static void deriv(const double s[4], const double tau[2], double damping, double d[4])
{
    double ddq[2];
    accel(s, tau, damping, ddq);
    d[0] = s[2]; d[1] = s[3]; d[2] = ddq[0]; d[3] = ddq[1];
}

static double wrap(double a)
{
    double r = fmod(a + PI, 2.0 * PI);
    if (r < 0.0)
        r += 2.0 * PI;
    return r - PI;
}

static double clampv(double v)
{
    return v < -MAX_VEL ? -MAX_VEL : (v > MAX_VEL ? MAX_VEL : v);
}

/* One env.step worth of integration: SUBSTEPS RK4 steps, then the velocity
 * clip and the angle wrap, in that order, exactly as env._integrate does. */
static void rk4_step(double s[4], const double tau[2], double damping)
{
    const double h = DT / SUBSTEPS;
    for (int n = 0; n < SUBSTEPS; n++) {
        double k1[4], k2[4], k3[4], k4[4], t[4];
        deriv(s, tau, damping, k1);
        for (int i = 0; i < 4; i++) t[i] = s[i] + 0.5 * h * k1[i];
        deriv(t, tau, damping, k2);
        for (int i = 0; i < 4; i++) t[i] = s[i] + 0.5 * h * k2[i];
        deriv(t, tau, damping, k3);
        for (int i = 0; i < 4; i++) t[i] = s[i] + h * k3[i];
        deriv(t, tau, damping, k4);
        for (int i = 0; i < 4; i++)
            s[i] = s[i] + (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
    s[2] = clampv(s[2]);
    s[3] = clampv(s[3]);
    s[0] = wrap(s[0]);
    s[1] = wrap(s[1]);
}

/* The integrator the README says would break the task, for comparison only. */
static void euler_step(double s[4], const double tau[2], double damping)
{
    const double h = DT / SUBSTEPS;
    for (int n = 0; n < SUBSTEPS; n++) {
        double ddq[2];
        accel(s, tau, damping, ddq);
        s[2] += h * ddq[0];
        s[3] += h * ddq[1];
        s[0] += h * s[2];
        s[1] += h * s[3];
    }
    s[2] = clampv(s[2]);
    s[3] = clampv(s[3]);
    s[0] = wrap(s[0]);
    s[1] = wrap(s[1]);
}

static double energy(const double s[4])
{
    double m[2][2];
    mass_matrix(s[1], m);
    return 0.5 * (s[2] * (m[0][0] * s[2] + m[0][1] * s[3])
                + s[3] * (m[1][0] * s[2] + m[1][1] * s[3]));
}

/* --- CSV reading, columns resolved by name so an added column cannot shift
 * what this reads out from under it. --- */
static int column_of(const char *header, const char *name)
{
    char buf[LINE];
    snprintf(buf, sizeof buf, "%s", header);
    int i = 0;
    for (char *tok = strtok(buf, ",\r\n"); tok; tok = strtok(NULL, ",\r\n"), i++)
        if (strcmp(tok, name) == 0)
            return i;
    return -1;
}

static const char *field(const char *line, int index)
{
    static char out[256];
    int col = 0;
    const char *p = line;
    while (col < index) {
        p = strchr(p, ',');
        if (!p) return NULL;
        p++; col++;
    }
    const char *end = strchr(p, ',');
    size_t n = end ? (size_t)(end - p) : strlen(p);
    if (n >= sizeof out) n = sizeof out - 1;
    memcpy(out, p, n);
    out[n] = '\0';
    char *nl = strpbrk(out, "\r\n");
    if (nl) *nl = '\0';
    return out;
}

typedef struct {
    char name[64];
    double damping, tau[2];
    int n;
    int step[MAX_ROWS];
    double s[MAX_ROWS][4];
    double energy[MAX_ROWS];
} Case;

int main(int argc, char **argv)
{
    const char *root = argc > 1 ? argv[1] : ".";
    char path[1024], line[LINE], header[LINE];

    snprintf(path, sizeof path, "%s/verify/golden/physics_trace.csv", root);
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 2; }
    if (!fgets(header, sizeof header, f)) { fclose(f); return 2; }

    const char *names[] = { "case", "damping", "tau1", "tau2", "step",
                            "q1", "q2", "dq1", "dq2", "energy" };
    int c[10];
    for (int i = 0; i < 10; i++) {
        c[i] = column_of(header, names[i]);
        if (c[i] < 0) {
            fprintf(stderr, "physics_trace.csv has no column %s\n", names[i]);
            fclose(f);
            return 2;
        }
    }

    Case cases[MAX_CASES];
    int n_cases = 0;
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '\n' || line[0] == '\0') continue;
        char name[64];
        /* snprintf rather than a bounded copy: gcc rejects the truncating form
         * under -Wstringop-truncation -Werror, and this one always terminates. */
        snprintf(name, sizeof name, "%s", field(line, c[0]));

        Case *k = NULL;
        for (int i = 0; i < n_cases; i++)
            if (strcmp(cases[i].name, name) == 0) k = &cases[i];
        if (!k) {
            if (n_cases == MAX_CASES) { fprintf(stderr, "too many cases\n"); return 2; }
            k = &cases[n_cases++];
            memset(k, 0, sizeof *k);
            snprintf(k->name, sizeof k->name, "%s", name);
            k->damping = atof(field(line, c[1]));
            k->tau[0] = atof(field(line, c[2]));
            k->tau[1] = atof(field(line, c[3]));
        }
        if (k->n == MAX_ROWS) { fprintf(stderr, "too many rows\n"); return 2; }
        const int r = k->n++;
        k->step[r] = atoi(field(line, c[4]));
        k->s[r][0] = atof(field(line, c[5]));
        k->s[r][1] = atof(field(line, c[6]));
        k->s[r][2] = atof(field(line, c[7]));
        k->s[r][3] = atof(field(line, c[8]));
        k->energy[r] = atof(field(line, c[9]));
    }
    fclose(f);

    if (n_cases == 0) { fprintf(stderr, "no cases in physics_trace.csv\n"); return 2; }

    int failures = 0;
    printf("reintegrating %d cases from verify/golden/physics_trace.csv\n", n_cases);
    for (int i = 0; i < n_cases; i++) {
        Case *k = &cases[i];
        double s[4] = { k->s[0][0], k->s[0][1], k->s[0][2], k->s[0][3] };
        int step = k->step[0];
        double worst = 0.0, worst_e = 0.0;

        for (int r = 1; r < k->n; r++) {
            while (step < k->step[r]) { rk4_step(s, k->tau, k->damping); step++; }
            for (int j = 0; j < 4; j++) {
                const double d = fabs(s[j] - k->s[r][j]);
                if (d > worst) worst = d;
            }
            const double de = fabs(energy(s) - k->energy[r]);
            if (de > worst_e) worst_e = de;
        }
        const int bad = worst > TOL || worst_e > TOL;
        failures += bad;
        printf("  %-18s %3d steps  worst |dstate| %.1e  worst |denergy| %.1e  %s\n",
               k->name, k->step[k->n - 1], worst, worst_e, bad ? "FAIL" : "ok");
    }

    /* The integrator claim, measured rather than asserted. Same initial state
     * as tests/test_physics.py, no damping, no torque, so total energy is a
     * constant of the motion and any drift is the integrator's own. */
    const double s0[4] = { 0.3, -0.8, 2.0, -1.5 };
    const double zero[2] = { 0.0, 0.0 };
    double a[4], b[4];
    memcpy(a, s0, sizeof a);
    memcpy(b, s0, sizeof b);
    const double e0 = energy(a);
    for (int i = 0; i < 500; i++) { rk4_step(a, zero, 0.0); euler_step(b, zero, 0.0); }
    const double rk4_drift = fabs(energy(a) - e0) / e0;
    const double euler_drift = fabs(energy(b) - e0) / e0;

    /* 1e-3 is the threshold tests/test_physics.py already demands of RK4.
     * The point of measuring Euler on the same trajectory is that the test is
     * only evidence for choosing RK4 if the alternative would actually fail
     * it, and nothing here had ever run the alternative. */
    printf("\nunforced arm, no damping, 500 steps at dt=%.2f, "
           "threshold 1e-3 from tests/test_physics.py\n", DT);
    printf("  RK4                  energy drift %.3e\n", rk4_drift);
    printf("  semi-implicit Euler  energy drift %.3e  (%.2gx worse)\n",
           euler_drift, euler_drift / rk4_drift);
    if (!(rk4_drift < 1e-3)) {
        printf("  FAIL: RK4 is not conserving energy here\n");
        failures++;
    }
    if (!(euler_drift > 1e-3)) {
        printf("  FAIL: Euler did not inject energy, so the README's reason "
               "for RK4 does not hold\n");
        failures++;
    }

    if (failures) {
        printf("\n%d checks failed\n", failures);
        return 1;
    }
    printf("\nC reproduces every logged state to %.0e and confirms the "
           "integrator claim\n", TOL);
    return 0;
}
