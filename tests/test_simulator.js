"use strict";

const assert = require("assert");
const Sim = require("../site/sim.js");

const projection = Sim.project({
  away:{team:"A",rating:-3,off:-1,def:-2},
  home:{team:"H",rating:5,off:4,def:3},
  homeField:2.4, homeScoringBump:1.2, leagueAverage:27.5
});
assert(Math.abs(projection.margin - 10.4) < 1e-9);
assert(Math.abs(projection.homePoints - 34.7) < 1e-9);
assert(Math.abs(projection.awayPoints - 23.5) < 1e-9);

const input = {
  away:{team:"A",rating:-3,off:-1,def:-2},
  home:{team:"H",rating:5,off:4,def:3},
  homeField:2.4, homeScoringBump:1.2, leagueAverage:27.5,
  marginSd:13, totalSd:10, keyNumbers:true,
  spread:-7, marketTotal:55, iterations:10000, seed:"offline-test"
};
const first = Sim.simulate(input);
const second = Sim.simulate(input);
assert.deepStrictEqual(first, second, "same inputs must be reproducible");
assert.strictEqual(first.iterations, 10000);
assert(first.homeWinProbability > .70 && first.homeWinProbability < .85);
assert(first.spread.homeCoverProbability > .50);
assert(first.total.overProbability > .50);
assert(Math.abs(first.homeWinProbability + first.awayWinProbability - 1) < 1e-12);
assert(Math.abs(first.spread.homeCoverProbability + first.spread.awayCoverProbability +
  first.spread.pushProbability - 1) < 1e-12);
assert(Math.abs(first.total.overProbability + first.total.underProbability +
  first.total.pushProbability - 1) < 1e-12);
assert(Sim.fairAmerican(.5) === -100);
console.log("simulator checks passed");
