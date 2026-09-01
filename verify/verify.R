# The inference behind the five-seed result, done properly and independently.
#
# reports/results.md publishes 43.2% +/- 5.8% over five seeds and the README
# draws a conclusion from it: that the spread is wide enough that "anything
# smaller than about 15 points is indistinguishable from a lucky seed". Both
# come from two calls to numpy in src/rlarm/report.py. Nothing recomputed the
# summary, and nothing tested the claim at all.
#
# This does three things, in base R with no packages:
#
#   1. recomputes the mean and the standard deviation from the five published
#      per-seed rates, which must match the published cells exactly
#   2. asks whether the five seeds differ by more than the binomial noise of
#      200 episodes each, which is the question "is the spread real"
#   3. computes the difference this design could actually detect, which is the
#      quantity the README's "about 15 points" is an estimate of
#
# Everything is read out of reports/results.md, the published document, not out
# of the JSON, so an edit to either one on its own is caught.

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) > 0) args[1] else "."
md <- readLines(file.path(root, "reports", "results.md"), warn = FALSE)

POINT_TOL <- 1e-12
ALPHA <- 0.05
POWER <- 0.80
failures <- 0

# --- read the published numbers -----------------------------------------
pull <- function(pattern, group = 2) {
    line <- grep(pattern, md, value = TRUE)
    if (length(line) != 1) stop(sprintf("expected one line matching %s, got %d",
                                        pattern, length(line)))
    m <- regmatches(line, regexec(pattern, line))[[1]]
    m[group]
}

n_ep <- as.numeric(pull("same ([0-9]+) held-out"))

seed_line <- grep("^Per-seed success rate:", md, value = TRUE)
if (length(seed_line) != 1) stop("no per-seed success line in results.md")
hits <- regmatches(seed_line, gregexpr("seed [0-9]+: [0-9.]+%", seed_line))[[1]]
rates <- as.numeric(sub(".*: ([0-9.]+)%", "\\1", hits)) / 100
if (length(rates) < 2) stop("fewer than two seeds in results.md")

row <- grep("^\\| success_rate \\|", md, value = TRUE)
cells <- trimws(strsplit(row, "\\|")[[1]])
published_mean <- as.numeric(sub("%", "", cells[3])) / 100
published_sd <- as.numeric(sub("%", "", cells[4])) / 100

cat(sprintf("reports/results.md: %d seeds, %g episodes each\n", length(rates), n_ep))
cat(sprintf("  per seed  %s\n", paste(sprintf("%.1f%%", 100 * rates), collapse = "  ")))

# --- 1. the published summary -------------------------------------------
# numpy.std defaults to ddof = 0, so this is the population divisor, n.
pop_sd <- function(x) sqrt(sum((x - mean(x))^2) / length(x))
got_mean <- mean(rates)
got_sd <- pop_sd(rates)
# The published cells are rounded to one decimal of a percent, so agreement is
# checked at that resolution rather than to machine precision.
dm <- abs(round(100 * got_mean, 1) - round(100 * published_mean, 1))
ds <- abs(round(100 * got_sd, 1) - round(100 * published_sd, 1))
ok <- dm <= POINT_TOL && ds <= POINT_TOL
failures <- failures + !ok
cat(sprintf("\nsummary recomputed from the per-seed rates\n"))
cat(sprintf("  mean %.1f%% against published %.1f%%   sd %.1f%% against published %.1f%%   %s\n",
            100 * got_mean, 100 * published_mean, 100 * got_sd, 100 * published_sd,
            if (ok) "ok" else "FAIL"))

# --- 2. is the seed spread bigger than 200-episode noise? ----------------
# Each seed is 200 independent episodes, so even identical policies would not
# score identically. The question the README leans on is whether the seeds
# differ by more than that. A homogeneity test on the 5 x 2 table of successes
# and failures answers it. Nothing in the repository had asked.
successes <- round(rates * n_ep)
if (any(abs(rates * n_ep - successes) > 1e-9)) {
    cat("FAIL: a per-seed rate is not a whole number of episodes\n")
    failures <- failures + 1
}
tab <- cbind(successes, n_ep - successes)
test <- chisq.test(tab)
binom_sd <- sqrt(got_mean * (1 - got_mean) / n_ep)
cat("\nis the seed spread larger than the sampling noise of 200 episodes?\n")
cat(sprintf("  between-seed sd   %.4f\n", sd(rates)))
cat(sprintf("  binomial sd at p=%.3f, n=%g   %.4f\n", got_mean, n_ep, binom_sd))
cat(sprintf("  homogeneity across %d seeds: chi-square %.2f on %d df, p = %.4f\n",
            length(rates), test$statistic, test$parameter, test$p.value))
if (test$p.value >= ALPHA) {
    cat("  FAIL: the seeds are consistent with one policy, so the reported\n")
    cat("        spread would be sampling noise rather than seed variance\n")
    failures <- failures + 1
} else {
    cat(sprintf("  the seeds really do differ, p < %.2f, so the spread is seed\n", ALPHA))
    cat("  variance and not an artefact of the episode count\n")
}

# --- 3. what difference could five seeds detect? -------------------------
# The README says anything under about 15 points is indistinguishable from a
# lucky seed. That is a claim about the power of this design, so compute it:
# the smallest difference two five-seed runs could separate at 5% and 80%.
s <- sd(rates)
n <- length(rates)
df <- 2 * n - 2
mdd <- (qt(1 - ALPHA / 2, df) + qt(POWER, df)) * s * sqrt(2 / n)
half <- qt(1 - ALPHA / 2, n - 1) * s / sqrt(n)
cat("\nwhat this design can resolve\n")
cat(sprintf("  95%% interval on the mean          %.1f%% +/- %.1f points\n",
            100 * got_mean, 100 * half))
cat(sprintf("  detectable difference, %d vs %d seeds, alpha %.2f, power %.2f   %.1f points\n",
            n, n, ALPHA, POWER, 100 * mdd))
# The README rounds this to "about 15 points". Allow five points of slack in
# either direction: the point is that the stated resolution follows from the
# data, so halving the observed spread would fail this.
claimed <- 15
if (abs(100 * mdd - claimed) > 5) {
    cat(sprintf("  FAIL: the README says about %d points, the data says %.1f\n",
                claimed, 100 * mdd))
    failures <- failures + 1
} else {
    cat(sprintf("  the README's \"about %d points\" is within 5 points of that\n", claimed))
}

if (failures > 0) {
    cat(sprintf("\n%d checks failed\n", failures))
    quit(status = 1)
}
cat("\nR reproduces the published summary and confirms the spread is real\n")
