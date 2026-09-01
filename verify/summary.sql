-- Rebuild both tables of reports/results.md from reports/results.json.
--
-- reports/results.md and the results table in the README are written by
-- src/rlarm/report.py from reports/results.json. The five-seed row is an
-- aggregation: report.py takes the mean and the population standard deviation
-- over the five per-seed records and prints them. Nothing checked either the
-- aggregation or the formatting, because the markdown and the JSON come out of
-- the same function in the same pass.
--
-- This derives every published cell from the per-seed and per-version records
-- in SQL, formatted exactly as report.py formats it, so verify/verify.sh can
-- diff the two texts. Percentages are Python's ".1%", distances ".3f", episode
-- lengths ".1f", the two diagnostics ".2f" and ".1f".
--
-- Run: sqlite3 -init verify/summary.sql :memory: ""

.mode list
.headers off

CREATE TEMP TABLE doc AS SELECT readfile('reports/results.json') AS j;

-- One row per reward version, in the order report.py writes them.
CREATE TEMP TABLE versions (ord INTEGER, name TEXT);
INSERT INTO versions VALUES
    (0, 'random'), (1, 'v1_sparse'), (2, 'v2_distance'), (3, 'v3_penalties'),
    (4, 'v4_potential'), (5, 'v5_progress'), (6, 'v6_goalfocus');

CREATE TEMP VIEW metrics AS
SELECT v.ord, v.name,
       json_extract(d.j, '$.' || v.name || '.success_rate')        AS success_rate,
       json_extract(d.j, '$.' || v.name || '.collision_rate')      AS collision_rate,
       json_extract(d.j, '$.' || v.name || '.timeout_rate')        AS timeout_rate,
       json_extract(d.j, '$.' || v.name || '.mean_final_dist')     AS mean_final_dist,
       json_extract(d.j, '$.' || v.name || '.mean_len')            AS mean_len,
       json_extract(d.j, '$.' || v.name || '.reached_target_rate') AS reached_target_rate,
       json_extract(d.j, '$.' || v.name || '.settle_given_reached') AS settle_given_reached,
       json_extract(d.j, '$.' || v.name || '.mean_speed_near')     AS mean_speed_near,
       json_extract(d.j, '$.' || v.name || '.mean_collision_step') AS mean_collision_step
FROM versions v, doc d;

SELECT printf('| %s | %.1f%% | %.1f%% | %.1f%% | %.3f | %.1f | %.1f%% | %.1f%% | %.2f | %.1f |',
              name, 100 * success_rate, 100 * collision_rate, 100 * timeout_rate,
              mean_final_dist, mean_len, 100 * reached_target_rate,
              100 * settle_given_reached, mean_speed_near, mean_collision_step)
FROM metrics ORDER BY ord;

-- The five-seed table. Mean and standard deviation are recomputed here from the
-- per_seed records rather than read from the mean and std that report.py
-- stored, which is the only way this can catch a mistake in the aggregation.
-- numpy's std defaults to ddof=0, so the divisor is n and not n-1.
CREATE TEMP TABLE per_seed AS
SELECT json_extract(e.value, '$.seed')                  AS seed,
       json_extract(e.value, '$.success_rate')          AS success_rate,
       json_extract(e.value, '$.collision_rate')        AS collision_rate,
       json_extract(e.value, '$.timeout_rate')          AS timeout_rate,
       json_extract(e.value, '$.mean_final_dist')       AS mean_final_dist,
       json_extract(e.value, '$.mean_len')              AS mean_len
FROM doc d, json_each(json_extract(d.j, '$.final_multiseed.per_seed')) e;

CREATE TEMP VIEW long AS
    SELECT 0 AS ord, 'success_rate'    AS metric, success_rate    AS x, 1 AS pct FROM per_seed
    UNION ALL SELECT 1, 'collision_rate',   collision_rate,   1 FROM per_seed
    UNION ALL SELECT 2, 'timeout_rate',     timeout_rate,     1 FROM per_seed
    UNION ALL SELECT 3, 'mean_final_dist',  mean_final_dist,  0 FROM per_seed
    UNION ALL SELECT 4, 'mean_len',         mean_len,         0 FROM per_seed;

CREATE TEMP VIEW agg AS
SELECT ord, metric, pct, AVG(x) AS mean, MIN(x) AS lo, MAX(x) AS hi, COUNT(*) AS n
FROM long GROUP BY ord, metric, pct;

SELECT printf('| %s | %.1f%% | %.1f%% | %.1f%% | %.1f%% |',
              a.metric, 100 * a.mean, 100 * s.sd, 100 * a.lo, 100 * a.hi)
FROM agg a
JOIN (SELECT l.ord, sqrt(SUM((l.x - a.mean) * (l.x - a.mean)) / a.n) AS sd
      FROM long l JOIN agg a ON a.ord = l.ord GROUP BY l.ord) s ON s.ord = a.ord
WHERE a.pct = 1
UNION ALL
SELECT printf('| %s | %.3f | %.3f | %.3f | %.3f |',
              a.metric, a.mean, s.sd, a.lo, a.hi)
FROM agg a
JOIN (SELECT l.ord, sqrt(SUM((l.x - a.mean) * (l.x - a.mean)) / a.n) AS sd
      FROM long l JOIN agg a ON a.ord = l.ord GROUP BY l.ord) s ON s.ord = a.ord
WHERE a.pct = 0;
