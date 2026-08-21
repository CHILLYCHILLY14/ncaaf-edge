# Setup

Start to finish: about 30 minutes, of which 20 is the first backfill running by
itself. Everything here is free — GitHub Actions and GitHub Pages both have free
tiers this fits inside comfortably, and ESPN's data needs no key.

---

## 1. Put the code on GitHub

Create a new repository (call it `ncaaf-edge`), then from the unzipped folder:

```bash
cd ncaaf-edge
git init
git add -A
git commit -m "NCAAF edge model"
git branch -M main
git remote add origin https://github.com/YOURNAME/ncaaf-edge.git
git push -u origin main
```

**Public or private?** Read step 6 before deciding — it's the one real trade-off
in this setup.

---

## 2. Run it once locally

Worth doing so the first backfill isn't happening blind inside a CI log.

The download ships with **simulated data** in `site/data/` so the dashboard and
the workbook show something the moment you open them. Both say so on their face
— an amber banner on the page, a note on the workbook's Read Me sheet. The first
real build overwrites it.

```bash
pip install -r requirements.txt
python -m tests.test_offline      # ~90 checks, no network
python -m pipeline.build --full   # backfills last season, 5-20 min
python -m pipeline.to_excel
```

The first run walks ESPN day by day across two seasons, which is why it's slow.
Every run after that touches a rolling two-week window and takes seconds.

Open `site/index.html` in a browser and you should see the dashboard with real
numbers. If it says it can't read the data files, serve it over HTTP instead —
browsers block `fetch` from `file://`:

```bash
cd site && python -m http.server 8000    # then open http://localhost:8000
```

Commit what the run produced:

```bash
git add -A && git commit -m "first build" && git push
```

---

## 3. Turn on GitHub Pages

In the repo: **Settings → Pages → Build and deployment → Source: GitHub Actions.**

That's the whole configuration. The included workflow publishes the `site/`
folder every time it runs.

---

## 4. Let the schedule take over

**Settings → Actions → General → Workflow permissions → Read and write
permissions → Save.** Without this the job can't commit the ledger back.

The workflow (`.github/workflows/update.yml`) then runs at 9am, 1pm and 5pm ET
to refresh odds and edges, and at 11pm ET to grade the day's finals. Change the
cron lines if you want different times — they're in UTC.

To run it by hand: **Actions → Refresh model → Run workflow.** Tick the "full
backfill" box only if you need to rebuild the cache from scratch.

Your live dashboard is now at:

```
https://YOURNAME.github.io/ncaaf-edge/
```

and the JSON feed behind it at `https://YOURNAME.github.io/ncaaf-edge/data/`.

---

## 5. Embed it in Wix

In the Wix editor:

1. **Add Elements → Embed Code → Popular Embeds → Embed a Site.**
2. Drop it on the page and drag it wide — 1200px+ if your page allows, and as
   tall as you can. The embed is an iframe and won't auto-size itself.
3. Click **Enter Website Address** and paste your GitHub Pages URL.
4. **Apply.**

That's it — no Premium plan needed for embedding, and no Velo.

A few things worth knowing:

* **HTTPS only.** GitHub Pages is HTTPS, so you're fine.
* **The iframe won't grow with its content.** Set the element tall (1400px is
  about right for the Best Bets tab) and let the dashboard scroll inside it.
  It already posts its height to the parent window, so if you later switch to
  Velo you can auto-size it with `$w('#html1').onMessage(e => ...)`.
* **Switch to the mobile editor** and resize the embed there separately. Wix
  keeps mobile layout separate, and the dashboard is responsive down to phone
  width.
* If you'd rather point the embed at a different feed, the page accepts
  `?data=https://.../data/` and will read from there instead.

---

## 6. Keeping it private

This is the honest part, because "private" and "free" pull against each other.

**What Wix can do:** restrict the page to logged-in members. Add the Members
Area app, then in the Pages menu → your page → **Permissions → Members only**.
Now only you (and anyone you invite) can load that page.

**What that does not do:** hide the JSON. GitHub Pages is public hosting — the
site is served to anyone who requests it, whether or not the repo is private.
Someone who knows or guesses `YOURNAME.github.io/ncaaf-edge/data/board.json` can
read your board. There is no free way around that, and a client-side password
would only be theatre, since the browser has to fetch the data either way.

How much that matters is your call — it's your model's opinions, not your bank
details. Three options, in order of how much privacy they buy:

| | Privacy | Effort |
|---|---|---|
| Gate the Wix page, accept the feed is reachable | Page is private, data is discoverable | What you just did |
| Publish under an unguessable path, e.g. `site/` → `site/x7k2m9/` | Obscurity only, but real in practice | 5 min |
| Never publish: private repo, run `python -m pipeline.build` locally, open `site/index.html` | Complete | No Wix, no schedule |

For a personal model, gating the Wix page and not advertising the feed URL is a
reasonable place to land. Just don't mistake it for real access control.

---

## 7. Tune it

`config/settings.json` is the whole control panel. Edit, commit, push, and the
next run uses it.

The three you'll actually touch:

* **`tiers`** — the edge required for BEST BET / GOOD / LEAN.
* **`market_blend`** — how much you defer to the market. Higher means fewer,
  more conservative bets. `0.50` is the default because it minimised
  out-of-sample calibration error in walk-forward simulation.
* **`filters.max_plays_per_week`** — volume cap, best edges first.

Before changing anything, run the backtest so you're tuning against evidence:

```bash
python -m pipeline.backtest --season 2025
```

---

## 8. Before you bet real money

Let it run for a few weeks with `"max_plays_per_week": 0` in the filters — it
prices and logs everything without you staking anything — and watch the Model
Health tab. Specifically:

* Is the calibration gap roughly centred on zero, or negative everywhere?
* Is average CLV above zero?
* Is BEST BET out-earning LEAN, or the other way round?

A model that fails those isn't unlucky, it's wrong, and the tiers on the front
page are decoration until it passes them.

---

## Troubleshooting

**Workflow fails on the commit step.** Workflow permissions aren't set to read
and write — step 4.

**Dashboard says it can't read the data files.** Either the first build hasn't
run, or you opened `index.html` from `file://`. Serve it over HTTP.

**Board is empty in August.** Correct. Odds don't post until the week of the
game, and the model won't invent a line it can't see.

**Every game says PASS.** Also correct, and common. The model agreeing with the
market is the normal state.

**Ratings all sit near zero in week 1.** By design — with almost no evidence,
the honest estimate is "average", and the ridge penalty enforces that.

**Odds vanished from a game that just finished.** ESPN drops them at final. The
pipeline snapshots every line it sees into `state/lines.json` for exactly this
reason, which is why that folder must stay committed.
