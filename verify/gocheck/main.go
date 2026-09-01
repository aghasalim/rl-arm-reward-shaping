// Structural validation of every results file, plus an independent
// recomputation of the five-seed aggregate and a cell by cell check of the
// README table against the JSON it was written from.
//
// reports/results.json is the evidence for the whole results section. Nothing
// checked that it is well formed: a rate outside [0, 1], a NaN that leaked out
// of a division over zero collisions, an episode count that quietly changed,
// or a README table edited by hand after the JSON moved would all be invisible
// until a reader worked it out for themselves.
//
// Run: cd verify/gocheck && go run . -root ..
package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

const tol = 1e-12

// The columns of the results table in README.md, in order, and where each one
// comes from in reports/results.json.
var readmeRows = []struct {
	label string
	key   string
}{
	{"random policy", "random"},
	{"v1 sparse", "v1_sparse"},
	{"v2 distance", "v2_distance"},
	{"v3 penalties", "v3_penalties"},
	{"v4 potential", "v4_potential"},
	{"v5 progress", "v5_progress"},
	{"v6 goal-focus", "v6_goalfocus"},
}

type metrics map[string]float64

var problems []string

func fail(format string, a ...any) {
	problems = append(problems, fmt.Sprintf(format, a...))
}

func pct(x float64) string { return fmt.Sprintf("%.1f%%", 100*x) }

// --- structural checks ---------------------------------------------------

func checkCSV(path string, wantRows int) {
	f, err := os.Open(path)
	if err != nil {
		fail("%s: %v", filepath.Base(path), err)
		return
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.FieldsPerRecord = 0 // a ragged file is an error, which is the point
	rows, err := r.ReadAll()
	if err != nil {
		fail("%s: %v", filepath.Base(path), err)
		return
	}
	if len(rows) < 2 {
		fail("%s: only %d rows", filepath.Base(path), len(rows))
		return
	}

	seen := map[string]bool{}
	for _, h := range rows[0] {
		if strings.TrimSpace(h) == "" {
			fail("%s: a column has an empty name", filepath.Base(path))
		}
		if seen[h] {
			fail("%s: duplicate column %q", filepath.Base(path), h)
		}
		seen[h] = true
	}
	for i, row := range rows[1:] {
		for j, cell := range row {
			low := strings.ToLower(strings.TrimSpace(cell))
			if low == "nan" || low == "inf" || low == "-inf" || low == "+inf" {
				fail("%s: row %d column %s is %s", filepath.Base(path), i+2, rows[0][j], cell)
			}
		}
	}
	if wantRows > 0 && len(rows)-1 != wantRows {
		fail("%s: %d data rows, expected %d", filepath.Base(path), len(rows)-1, wantRows)
	}
	fmt.Printf("  %-22s %5d rows, %2d columns, no ragged rows, no NaN or Inf\n",
		filepath.Base(path), len(rows)-1, len(rows[0]))
}

// checkRates enforces what a rate is allowed to be, and the identity the
// evaluator produces it with: every episode ends in success, collision or the
// step limit, so the three rates sum to one.
func checkRates(what string, m metrics, wantEpisodes float64) {
	for k, v := range m {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			fail("%s: %s is %v", what, k, v)
		}
		if strings.HasSuffix(k, "_rate") && (v < 0 || v > 1) {
			fail("%s: %s is %g, outside [0, 1]", what, k, v)
		}
	}
	if n, ok := m["n_episodes"]; ok && n != wantEpisodes {
		fail("%s: n_episodes is %g, expected %g", what, n, wantEpisodes)
	}
	sum := m["success_rate"] + m["collision_rate"] + m["timeout_rate"]
	if math.Abs(sum-1.0) > 1e-9 {
		fail("%s: success + collision + timeout is %.12f, not 1", what, sum)
	}
	// A rate over n episodes has to be a whole number of episodes.
	if n, ok := m["n_episodes"]; ok {
		for _, k := range []string{"success_rate", "collision_rate", "reached_target_rate"} {
			c := m[k] * n
			if math.Abs(c-math.Round(c)) > 1e-9 {
				fail("%s: %s is %g, which is not a whole number of %g episodes", what, k, m[k], n)
			}
		}
	}
}

func mean(xs []float64) float64 {
	s := 0.0
	for _, x := range xs {
		s += x
	}
	return s / float64(len(xs))
}

// Population standard deviation, the ddof=0 that numpy.std defaults to and that
// report.py therefore published.
func stdev(xs []float64) float64 {
	m := mean(xs)
	s := 0.0
	for _, x := range xs {
		s += (x - m) * (x - m)
	}
	return math.Sqrt(s / float64(len(xs)))
}

func main() {
	root := flag.String("root", ".", "repository root")
	flag.Parse()

	fmt.Println("structural validation")
	checkCSV(filepath.Join(*root, "verify", "golden", "oracle_layouts.csv"), 400)
	checkCSV(filepath.Join(*root, "verify", "golden", "physics_trace.csv"), 0)
	checkCSV(filepath.Join(*root, "verify", "golden", "reward_trace.csv"), 0)

	// Python writes NaN and Infinity as bare words, which is not valid JSON.
	// Unmarshal rejects them, so this both parses the file and rules them out.
	for _, name := range []string{"results.json", "sweep.json"} {
		raw, err := os.ReadFile(filepath.Join(*root, "reports", name))
		if err != nil {
			fail("%s: %v", name, err)
			continue
		}
		var parsed any
		if err := json.Unmarshal(raw, &parsed); err != nil {
			fail("%s: %v", name, err)
			continue
		}
		fmt.Printf("  %-22s %5d bytes, valid JSON, no NaN or Infinity\n", name, len(raw))
	}

	raw, err := os.ReadFile(filepath.Join(*root, "reports", "results.json"))
	if err != nil {
		fmt.Fprintf(os.Stderr, "cannot read reports/results.json: %v\n", err)
		os.Exit(2)
	}
	var doc map[string]json.RawMessage
	if err := json.Unmarshal(raw, &doc); err != nil {
		fmt.Fprintf(os.Stderr, "reports/results.json: %v\n", err)
		os.Exit(2)
	}

	fmt.Println("\nper version records")
	for _, r := range readmeRows {
		var m metrics
		if err := json.Unmarshal(doc[r.key], &m); err != nil {
			fail("%s: %v", r.key, err)
			continue
		}
		checkRates(r.key, m, 200)
		fmt.Printf("  %-16s success %s  collision %s  timeout %s  over %g episodes\n",
			r.key, pct(m["success_rate"]), pct(m["collision_rate"]),
			pct(m["timeout_rate"]), m["n_episodes"])
	}

	var sweep []map[string]any
	if err := json.Unmarshal(mustRead(filepath.Join(*root, "reports", "sweep.json")), &sweep); err != nil {
		fail("sweep.json: %v", err)
	} else {
		for _, arm := range sweep {
			m := metrics{}
			for k, v := range arm {
				if f, ok := v.(float64); ok {
					m[k] = f
				}
			}
			checkRates(fmt.Sprintf("sweep arm %v", arm["arm"]), m, 150)
		}
		fmt.Printf("  %d sweep arms, all 150 episodes, all rates in range\n", len(sweep))
	}

	// --- the five-seed aggregate, recomputed ----------------------------
	var ms struct {
		PerSeed []metrics `json:"per_seed"`
		Mean    metrics   `json:"mean"`
		Std     metrics   `json:"std"`
		NSeeds  int       `json:"n_seeds"`
	}
	if err := json.Unmarshal(doc["final_multiseed"], &ms); err != nil {
		fmt.Fprintf(os.Stderr, "final_multiseed: %v\n", err)
		os.Exit(2)
	}
	if len(ms.PerSeed) != ms.NSeeds {
		fail("final_multiseed: %d per-seed records but n_seeds is %d", len(ms.PerSeed), ms.NSeeds)
	}

	fmt.Printf("\nrecomputing the %d-seed aggregate from the per-seed records\n", ms.NSeeds)
	keys := make([]string, 0, len(ms.Mean))
	for k := range ms.Mean {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		xs := make([]float64, 0, len(ms.PerSeed))
		for i, s := range ms.PerSeed {
			checkRates(fmt.Sprintf("seed %g", ms.PerSeed[i]["seed"]), s, 200)
			xs = append(xs, s[k])
		}
		dm, ds := math.Abs(mean(xs)-ms.Mean[k]), math.Abs(stdev(xs)-ms.Std[k])
		status := "ok"
		if dm > tol || ds > tol {
			status = "FAIL"
			fail("final_multiseed.%s: mean off by %.2e, std off by %.2e", k, dm, ds)
		}
		fmt.Printf("  %-22s mean %12.6f |d| %.1e   std %10.6f |d| %.1e   %s\n",
			k, mean(xs), dm, stdev(xs), ds, status)
	}

	// --- the README table, cell by cell ---------------------------------
	readme := string(mustRead(filepath.Join(*root, "README.md")))
	fmt.Println("\nREADME.md results table against reports/results.json")
	for _, r := range readmeRows {
		var m metrics
		json.Unmarshal(doc[r.key], &m)
		want := []string{
			pct(m["success_rate"]), pct(m["collision_rate"]), pct(m["timeout_rate"]),
			pct(m["reached_target_rate"]), pct(m["settle_given_reached"]),
			fmt.Sprintf("%.3f m", m["mean_final_dist"]),
		}
		checkRow(readme, r.label, want)
	}
	checkRow(readme, "**v6 goal-focus** (5 seeds)", []string{
		fmt.Sprintf("%s ± %s", pct(ms.Mean["success_rate"]), pct(ms.Std["success_rate"])),
		pct(ms.Mean["collision_rate"]), pct(ms.Mean["timeout_rate"]),
		pct(ms.Mean["reached_target_rate"]), pct(ms.Mean["settle_given_reached"]),
		fmt.Sprintf("%.3f m", ms.Mean["mean_final_dist"]),
	})

	// The Limitations section lists the five seed scores in ascending order.
	rates := make([]float64, 0, len(ms.PerSeed))
	for _, s := range ms.PerSeed {
		rates = append(rates, s["success_rate"])
	}
	sort.Float64s(rates)
	parts := make([]string, len(rates))
	for i, x := range rates {
		parts[i] = pct(x)
	}
	sentence := strings.Join(parts[:len(parts)-1], ", ") + " and " + parts[len(parts)-1]
	// The list is wrapped across lines in the prose, so compare on a copy with
	// runs of whitespace collapsed.
	if strings.Contains(strings.Join(strings.Fields(readme), " "), sentence) {
		fmt.Printf("  %-30s %s\n", "per-seed list", sentence)
	} else {
		fail("README.md does not contain the per-seed list %q", sentence)
	}

	if len(problems) > 0 {
		fmt.Printf("\n%d problems\n", len(problems))
		for _, p := range problems {
			fmt.Printf("  %s\n", p)
		}
		os.Exit(1)
	}
	fmt.Println("\nGo agrees with the published aggregate and every README cell, " +
		"and the results files are well formed")
}

// checkRow finds the markdown row whose first cell is label and requires each
// wanted value to appear in the cells after it, in order. Bold markers are
// stripped, so **100.0%** and 100.0% are the same cell.
func checkRow(readme, label string, want []string) {
	for _, line := range strings.Split(readme, "\n") {
		if !strings.HasPrefix(line, "| "+label+" |") {
			continue
		}
		cells := strings.Split(strings.ReplaceAll(strings.ReplaceAll(line, "**", ""), "*", ""), "|")
		for i := range cells {
			cells[i] = strings.TrimSpace(cells[i])
		}
		missing := []string{}
		at := 0
		for _, w := range want {
			found := false
			for ; at < len(cells); at++ {
				if cells[at] == w {
					found = true
					at++
					break
				}
			}
			if !found {
				missing = append(missing, w)
			}
		}
		if len(missing) > 0 {
			fail("README row %q is missing %v", label, missing)
			return
		}
		fmt.Printf("  %-30s %s\n", label, strings.Join(want, "  "))
		return
	}
	fail("README.md has no table row for %q", label)
}

func mustRead(path string) []byte {
	b, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "cannot read %s: %v\n", path, err)
		os.Exit(2)
	}
	return b
}
