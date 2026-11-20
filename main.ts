// gta6-trophy-rarity-forecast-01 entry: magnitude-weighted comparable-ratio
// ensemble over log-time trophy curves, with a dilution-aware spread.
//
// The shape of the problem (what the data says):
//   - A trophy's earn rate is earners / players-tracked. Both grow, on
//     different clocks: earners grow as the tracked players play through the
//     story; players-tracked grows in steps, each Exophase refresh adding a
//     batch of fresh players who hold almost nothing. So a rate's path is a
//     slow climb cut by sharp dilution drops at every refresh (launch night
//     saw one: 638 -> 3,868 tracked, every rate but the 100% one lower).
//   - In logit space the climb is close to linear in ln(1 + hours since
//     launch): fast early, flattening later. That is the house curve
//     baseline's form, and it is the right one — the work is in estimating
//     each trophy's line well and in spreading honestly around it.
//
// The forecast, per trophy, is built the way the gta6-twitch-ccu-forecast-01
// leaderboard winner was built (magnitude-weighted comparable-ratio
// ensemble), adapted from launches to trophies:
//   1. Fit, per trophy, a weighted least-squares line of logit(rate) on
//      ln(1 + hours since launch) over the case's history. Weights are
//      recency times earner count: a read where a trophy has many earners
//      pins its logit far tighter than one where it has four.
//   2. Shrink each slope toward its tier's precision-weighted mean slope
//      (platinum / gold / silver / bronze grow on different clocks). A rare
//      trophy's own line is noisy, so it borrows its tier's.
//   3. Blend the trophy's own predicted logit change to the target with the
//      magnitude-weighted geometric mean of the predicted changes of its
//      comparables — the trophies whose current rate sits within ~2 logit
//      units of its own (Gaussian kernel, bandwidth 1.0, as in the CCU
//      entry). A trophy with few earners leans on its comparables; one with
//      many trusts its own line.
//   4. Spread, in logit units, combines four honest sources: the random
//      walk over the horizon (the house baseline's 0.5 per sqrt-day), the
//      trophy's own fit residual, the disagreement among its comparables
//      (their weighted IQR / 1.349), and refresh risk — the typical logit
//      drop a refresh causes, measured from the history's own
//      players-tracked jumps, or a prior that scales with how small the
//      tracked base still is when no jump is visible (launch night's flat
//      history hides the refresh the practice is scored on).
//
// Node 22, standard library only, no network, deterministic: no clocks, no
// randomness; every sum walks trophies and reads in file order, so two runs
// on the same input write identical bytes.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Launch reference: the house curve baseline measures hours from here. */
const LAUNCH_MS = Date.parse('2026-11-19T00:00:00Z');
/** Magnitude kernel bandwidth, in logit units of current rate. */
const BANDWIDTH = 1.0;
/** Comparables are trophies within this many logit units of the target's rate. */
const COMP_WINDOW = 2.0;
/** Random-walk spread, in logit units per square-root day. */
const SIGMA_RW = 0.5;
/** Earner count at which a trophy's own line gets half the blend weight. */
const BLEND_EARNERS = 30;
/** Recency half-life, in hours, for the line fit. */
const RECENCY_HALF_LIFE_H = 24;
/** A fit's residual spread never goes below this, in logit units. */
const FIT_SPREAD_FLOOR = 0.1;
/** A fit's residual spread never goes above this. */
const FIT_SPREAD_CAP = 1.2;
/** Refresh-risk prior floor, in logit units. */
const REFRESH_PRIOR_FLOOR = 0.25;

interface Trophy {
  id: number;
  name: string;
  description: string;
  tier: string;
  points: number;
  secret: boolean;
}

interface Read {
  slot: string;
  at: string;
  players_tracked: number;
  trophies_earned: number;
  rates: Array<number | null>;
}

interface CaseFile {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  game: { name: string; platform: string; release_date: string; source: string };
  trophies: Trophy[];
  history: Read[];
  notes?: string;
}

const clampPct = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10000) / 10000;
/** Hours from the launch reference to an ISO timestamp. */
const hoursSinceLaunch = (iso: string): number =>
  Math.max(0, (Date.parse(iso) - LAUNCH_MS) / 3_600_000);

/** Median of numbers, in input order for a deterministic tie. */
const median = (xs: number[]): number => {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 === 1 ? s[m] : (s[m - 1] + s[m]) / 2;
};

/** Weighted mean of (value, weight) pairs. */
const weightedMean = (pairs: Array<[number, number]>): number => {
  let sw = 0;
  let sv = 0;
  for (const [v, w] of pairs) {
    sw += w;
    sv += w * v;
  }
  return sw > 0 ? sv / sw : 0;
};

/** Weighted quantile of (value, weight) pairs, by cumulative weight. */
const weightedQuantile = (pairs: Array<[number, number]>, p: number): number => {
  const sorted = [...pairs].sort((a, b) => a[0] - b[0]);
  const total = sorted.reduce((s, [, w]) => s + w, 0);
  if (total <= 0 || sorted.length === 0) return 0;
  let acc = 0;
  for (const [v, w] of sorted) {
    acc += w;
    if (acc >= p * total) return v;
  }
  return sorted[sorted.length - 1][0];
};

interface LineFit {
  /** Intercept at ln(1 + hours) = 0. */
  a: number;
  /** Slope per unit of ln(1 + hours). */
  b: number;
  /** Standard error of the slope (Infinity when unidentified). */
  se: number;
  /** Residual standard deviation, in logit units. */
  residSd: number;
  /** Number of reads used. */
  n: number;
}

/** Weighted least squares of y on x. Returns null when it cannot be fit. */
const fitLine = (xs: number[], ys: number[], ws: number[]): LineFit | null => {
  const n = xs.length;
  if (n < 2) return null;
  let sw = 0;
  let sx = 0;
  let sy = 0;
  for (let i = 0; i < n; i++) {
    sw += ws[i];
    sx += ws[i] * xs[i];
    sy += ws[i] * ys[i];
  }
  if (sw <= 0) return null;
  const xbar = sx / sw;
  const ybar = sy / sw;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - xbar;
    sxx += ws[i] * dx * dx;
    sxy += ws[i] * dx * (ys[i] - ybar);
  }
  if (sxx <= 1e-12) return { a: ybar, b: 0, se: Infinity, residSd: 0, n };
  const b = sxy / sxx;
  const a = ybar - b * xbar;
  let rss = 0;
  for (let i = 0; i < n; i++) {
    const r = ys[i] - (a + b * xs[i]);
    rss += ws[i] * r * r;
  }
  const dof = Math.max(1, n - 2);
  const residVar = rss / (sw * dof / n);
  const residSd = Math.sqrt(Math.max(0, residVar));
  const se = Math.sqrt(Math.max(0, residVar) / sxx);
  return { a, b, se, residSd, n };
};

/** Forecast one case: five quantiles per trophy, in trophy order. */
const forecastCase = (c: CaseFile): Array<{ id: number; q: number[] }> => {
  const last = c.history[c.history.length - 1];
  if (last === undefined) throw new Error(`${c.id} has no history`);
  const nT = c.trophies.length;
  const lastMs = Date.parse(last.at);
  const targetH = hoursSinceLaunch(c.target);
  const xTarget = Math.log(1 + targetH);
  const horizonDays = c.horizon_minutes / 1440;

  // Per-trophy series: log-time x, logit-rate y, weight, and the last rate.
  const lastRate: number[] = new Array(nT).fill(0);
  const lastEarners: number[] = new Array(nT).fill(0);
  const fits: Array<LineFit | null> = new Array(nT).fill(null);
  for (let i = 0; i < nT; i++) {
    const xs: number[] = [];
    const ys: number[] = [];
    const ws: number[] = [];
    let lr = 0;
    let le = 0;
    for (const read of c.history) {
      const r = read.rates[i];
      if (r === null || r === undefined) continue;
      const h = hoursSinceLaunch(read.at);
      const earners = Math.max(0, (r / 100) * read.players_tracked);
      const ageH = Math.max(0, (lastMs - Date.parse(read.at)) / 3_600_000);
      const recency = 0.25 + 0.75 * Math.pow(0.5, ageH / RECENCY_HALF_LIFE_H);
      const w = recency * Math.min(earners + 1, 100);
      xs.push(Math.log(1 + h));
      ys.push(logit(r));
      ws.push(w);
      lr = r;
      le = earners;
    }
    lastRate[i] = lr;
    lastEarners[i] = le;
    fits[i] = fitLine(xs, ys, ws);
  }

  // Tier shrinkage: each tier's precision-weighted mean slope, and the
  // cross-trophy spread of slopes inside the tier.
  const tierSlopes = new Map<string, { mean: number; tau: number }>();
  const tiers = [...new Set(c.trophies.map((t) => t.tier))];
  for (const tier of tiers) {
    const idx = c.trophies.map((t, i) => (t.tier === tier ? i : -1)).filter((i) => i >= 0);
    const pairs: Array<[number, number]> = [];
    const slopes: number[] = [];
    for (const i of idx) {
      const f = fits[i];
      if (f !== null && Number.isFinite(f.se) && f.se > 0) {
        pairs.push([f.b, 1 / (f.se * f.se)]);
        slopes.push(f.b);
      } else if (f !== null) {
        slopes.push(f.b);
      }
    }
    const mean = pairs.length > 0 ? weightedMean(pairs) : slopes.length > 0 ? median(slopes) : 0;
    let tau = 0.6;
    if (slopes.length >= 2) {
      const m = median(slopes);
      const devs = slopes.map((s) => Math.abs(s - m));
      tau = Math.min(1.5, Math.max(0.2, 1.4826 * median(devs)));
    }
    tierSlopes.set(tier, { mean, tau });
  }

  // Shrunk slope per trophy: precision-weighted blend of its own slope and
  // its tier's mean slope.
  const slope: number[] = new Array(nT).fill(0);
  for (let i = 0; i < nT; i++) {
    const f = fits[i];
    const tier = tierSlopes.get(c.trophies[i].tier) ?? { mean: 0, tau: 0.6 };
    if (f === null) {
      slope[i] = tier.mean;
    } else if (!Number.isFinite(f.se) || f.se <= 0) {
      slope[i] = tier.mean;
    } else {
      const pOwn = 1 / (f.se * f.se);
      const pTier = 1 / (tier.tau * tier.tau);
      slope[i] = (f.b * pOwn + tier.mean * pTier) / (pOwn + pTier);
    }
  }

  // Refresh risk: the typical logit drop a refresh causes, read off the
  // history's own players-tracked jumps; else a prior that scales with how
  // small the tracked base still is (a small base can multiply at a refresh).
  let refreshSpread = 0;
  {
    const drops: number[] = [];
    for (let j = 1; j < c.history.length; j++) {
      const prev = c.history[j - 1];
      const cur = c.history[j];
      if (cur.players_tracked > 1.2 * prev.players_tracked && prev.players_tracked > 0) {
        const d: number[] = [];
        for (let i = 0; i < nT; i++) {
          const a = prev.rates[i];
          const b = cur.rates[i];
          if (a !== null && a !== undefined && b !== null && b !== undefined && a > 0 && b > 0) {
            d.push(logit(a) - logit(b));
          }
        }
        if (d.length > 0) drops.push(median(d));
      }
    }
    if (drops.length > 0) {
      refreshSpread = Math.max(0.15, median(drops) * 0.5);
    } else {
      // No refresh visible in the history, so the risk is a prior: a refresh
      // dilutes every rate by ln(N_after / N_before), and when the tracked
      // base is still small a single refresh can multiply it several-fold
      // (launch night: 638 -> 3,868, a 1.8-logit drop). The prior therefore
      // grows as the base shrinks, and is nearly flat once the base is large
      // — the final's random walk already carries its uncertainty, so this
      // term barely moves a large-N forecast while covering a small-N one.
      const nLast = Math.max(100, last.players_tracked);
      refreshSpread = Math.min(
        1.2,
        Math.max(REFRESH_PRIOR_FLOOR, 1.1 * Math.pow(1000 / nLast, 0.25)),
      );
    }
  }

  // The last read's log-time, for the change from now to the target.
  const xLast = Math.log(1 + hoursSinceLaunch(last.at));
  const dx = xTarget - xLast;
  const rwSpread = SIGMA_RW * Math.sqrt(horizonDays);

  const rows: Array<{ id: number; q: number[] }> = [];
  for (let i = 0; i < nT; i++) {
    const ownDelta = slope[i] * dx;
    const ownLogit = logit(lastRate[i]);

    // Comparables: trophies of similar current magnitude. Their predicted
    // logit changes, weighted by magnitude similarity and reliability.
    const pairs: Array<[number, number]> = [];
    for (let j = 0; j < nT; j++) {
      if (j === i) continue;
      const dLogit = logit(lastRate[j]) - ownLogit;
      if (Math.abs(dLogit) > COMP_WINDOW) continue;
      const sim = Math.exp(-0.5 * Math.pow(dLogit / BANDWIDTH, 2));
      const rel = Math.min(lastEarners[j], 50) / 50;
      pairs.push([slope[j] * dx, sim * rel]);
    }
    const ensDelta = pairs.length > 0 ? weightedMean(pairs) : ownDelta;
    const iqr =
      pairs.length >= 4
        ? Math.max(0, weightedQuantile(pairs, 0.75) - weightedQuantile(pairs, 0.25)) / 1.349
        : 0;

    // Blend: a trophy with many earners trusts its own line; a rare one
    // leans on its comparables.
    const wOwn = lastEarners[i] / (lastEarners[i] + BLEND_EARNERS);
    const delta = wOwn * ownDelta + (1 - wOwn) * ensDelta;
    const medLogit = ownLogit + delta;

    // Spread: random walk + own fit residual + refresh risk + comparable
    // disagreement (only where the ensemble carries weight).
    const f = fits[i];
    const fitSpread =
      f === null
        ? 0.25
        : Math.min(FIT_SPREAD_CAP, Math.max(FIT_SPREAD_FLOOR, f.residSd));
    const ensSpread = (1 - wOwn) * iqr;
    const s = Math.sqrt(
      rwSpread * rwSpread + fitSpread * fitSpread + refreshSpread * refreshSpread + ensSpread * ensSpread,
    );

    const qs = Z.map((z) => round4(clampPct(invLogit(medLogit + z * s))));
    for (let j = 1; j < qs.length; j++) if (qs[j] < qs[j - 1]) qs[j] = qs[j - 1];
    rows.push({ id: c.trophies[i].id, q: qs });
  }
  return rows;
};

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseFile;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  const trophies = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
