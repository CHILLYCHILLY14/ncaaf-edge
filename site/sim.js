(function (root, factory) {
  "use strict";
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.NCAAFSim = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const KEY_BUMPS = {
    0:.55,1:1.15,2:1.05,3:2.35,4:1.20,5:.95,6:1.10,7:1.95,8:1.05,9:.90,
    10:1.55,11:1.10,12:.85,13:.95,14:1.60,15:.90,16:.85,17:1.45,18:.90,
    19:.85,20:.95,21:1.35,22:.85,23:.85,24:1.15,25:.85,27:.95,28:1.15,
    31:1.05,35:1.00
  };

  function finite(value, fallback) {
    const n = Number(value);
    return Number.isFinite(n) ? n : fallback;
  }

  function normalPdf(x, mean, sd) {
    const z = (x - mean) / sd;
    return Math.exp(-.5 * z * z) / (sd * Math.sqrt(2 * Math.PI));
  }

  function weightedDistribution(start, end, weight) {
    const rows = [];
    let total = 0;
    for (let value = start; value <= end; value += 1) {
      const p = Math.max(0, weight(value));
      total += p;
      rows.push({value, p});
    }
    let cumulative = 0;
    rows.forEach(row => {
      row.p = total > 0 ? row.p / total : 1 / rows.length;
      cumulative += row.p;
      row.cumulative = cumulative;
    });
    rows[rows.length - 1].cumulative = 1;
    return rows;
  }

  function marginDistribution(mean, sd, keyNumbers) {
    mean = finite(mean, 0);
    sd = Math.max(.1, finite(sd, 13));
    return weightedDistribution(-70, 70, margin =>
      normalPdf(margin, mean, sd) * (keyNumbers === false ? 1 : (KEY_BUMPS[Math.abs(margin)] || 1))
    );
  }

  function totalDistribution(mean, sd) {
    mean = finite(mean, 55);
    sd = Math.max(.1, finite(sd, 10));
    return weightedDistribution(0, 129, total => normalPdf(total, mean, sd));
  }

  function hashSeed(text) {
    let h = 2166136261;
    for (const ch of String(text)) {
      h ^= ch.charCodeAt(0);
      h = Math.imul(h, 16777619);
    }
    return h >>> 0;
  }

  function rng(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0;
      a = a + 0x6D2B79F5 | 0;
      let t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  function draw(dist, random) {
    const needle = random();
    let lo = 0, hi = dist.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (needle <= dist[mid].cumulative) hi = mid;
      else lo = mid + 1;
    }
    return dist[lo].value;
  }

  function project(input) {
    const away = input.away || {};
    const home = input.home || {};
    const neutral = Boolean(input.neutral);
    const hfa = neutral ? 0 : finite(input.homeField, 2.4);
    const homeBump = neutral ? 0 : finite(input.homeScoringBump, 1.2);
    const league = finite(input.leagueAverage, 27.5);
    const margin = finite(home.rating, 0) - finite(away.rating, 0) + hfa;
    const homePoints = league + finite(home.off, 0) - finite(away.def, 0) + homeBump;
    const awayPoints = league + finite(away.off, 0) - finite(home.def, 0);
    return {
      margin,
      total: homePoints + awayPoints,
      homePoints,
      awayPoints
    };
  }

  function fairAmerican(probability) {
    const p = Math.min(.999999, Math.max(.000001, finite(probability, .5)));
    return Math.round(p >= .5 ? -100 * p / (1 - p) : 100 * (1 - p) / p);
  }

  function simulate(input) {
    const projection = project(input);
    const iterations = Math.max(1000, Math.min(50000, Math.round(finite(input.iterations, 10000))));
    const spread = input.spread === "" || input.spread == null ? null : finite(input.spread, null);
    const marketTotal = input.marketTotal === "" || input.marketTotal == null ? null : finite(input.marketTotal, null);
    const marginDist = marginDistribution(projection.margin, finite(input.marginSd, 13), input.keyNumbers !== false);
    const totalDist = totalDistribution(projection.total, finite(input.totalSd, 10));
    const seed = hashSeed(input.seed || [input.away && input.away.team, input.home && input.home.team,
      projection.margin, projection.total, spread, marketTotal, iterations].join("|"));
    const random = rng(seed);
    let homeWins = 0, awayWins = 0;
    let homeCovers = 0, awayCovers = 0, spreadPushes = 0;
    let overs = 0, unders = 0, totalPushes = 0;
    let homeScoreSum = 0, awayScoreSum = 0;

    for (let i = 0; i < iterations; i += 1) {
      const margin = draw(marginDist, random);
      const total = draw(totalDist, random);
      if (margin > 0 || (margin === 0 && random() >= .5)) homeWins += 1;
      else awayWins += 1;
      if (spread !== null) {
        const adjusted = margin + spread;
        if (adjusted > 0) homeCovers += 1;
        else if (adjusted < 0) awayCovers += 1;
        else spreadPushes += 1;
      }
      if (marketTotal !== null) {
        if (total > marketTotal) overs += 1;
        else if (total < marketTotal) unders += 1;
        else totalPushes += 1;
      }
      let homeScore = Math.max(0, Math.round((total + margin) / 2));
      let awayScore = Math.max(0, total - homeScore);
      if (homeScore === 0 && awayScore === 0) awayScore = total;
      homeScoreSum += homeScore;
      awayScoreSum += awayScore;
    }

    const ratio = value => value / iterations;
    const homeWinProbability = ratio(homeWins);
    return {
      iterations,
      seed,
      projection,
      averageHomeScore: homeScoreSum / iterations,
      averageAwayScore: awayScoreSum / iterations,
      homeWinProbability,
      awayWinProbability: ratio(awayWins),
      fairHomeMoneyline: fairAmerican(homeWinProbability),
      fairAwayMoneyline: fairAmerican(ratio(awayWins)),
      spread: spread === null ? null : {
        line: spread,
        homeCoverProbability: ratio(homeCovers),
        awayCoverProbability: ratio(awayCovers),
        pushProbability: ratio(spreadPushes)
      },
      total: marketTotal === null ? null : {
        line: marketTotal,
        overProbability: ratio(overs),
        underProbability: ratio(unders),
        pushProbability: ratio(totalPushes)
      }
    };
  }

  return {project, simulate, marginDistribution, totalDistribution, fairAmerican, hashSeed};
});
