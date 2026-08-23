# Put This Update on GitHub (Very Simple Version)

Yes: you can replace the repository files with everything in the fixed folder,
commit it, and run the workflow again.

The important part is to upload **everything**, not only the `site` folder. The
odds repair also changes `pipeline`, `config`, `tests`, `state`, and `.github`.

## Browser-only steps

1. Download and unzip the fixed package.
2. Open <https://github.com/CHILLYCHILLY14/ncaaf-edge>.
3. Make sure the branch button says **main**.
4. Click **Add file**, then **Upload files**.
5. Drag all the files and folders from *inside* the unzipped folder into the
   upload box. Do not drag the outer folder itself, or GitHub may put the whole
   project inside one extra folder.
6. Wait until GitHub finishes listing the uploaded files.
7. At the bottom, choose **Commit directly to the main branch**.
8. Type `fix odds and add simulator` in the message box.
9. Click **Commit changes**.
10. Click the repository's **Actions** tab.
11. Open **Refresh model**, click **Run workflow**, leave **Full-season
    backfill** unchecked, and click the green **Run workflow** button.
12. Wait until **Refresh model** and **Deploy to GitHub Pages** both have green
    check marks.
13. Open <https://chillychilly14.github.io/ncaaf-edge/> and hard-refresh the
    page (`Ctrl+Shift+R` on Windows or `Cmd+Shift+R` on Mac).

You should now see a **Simulator** tab. The Model Health tab should say the odds
integrity is **VERIFIED**.

## What will look different

- The old pending ledger entries disappear because they were created from
  unverified placeholder `-110` prices.
- The current board may show zero plays. That is intentional when preseason
  ratings disagree too sharply with the market.
- The Full Board still shows every verified market and explains why a row was
  forced to PASS.

## If GitHub will not upload a folder

Use GitHub Desktop instead:

1. Clone `CHILLYCHILLY14/ncaaf-edge` in GitHub Desktop.
2. Copy everything from the fixed folder into that cloned folder and allow it
   to replace files with the same names.
3. In GitHub Desktop, enter `fix odds and add simulator` as the summary.
4. Click **Commit to main**, then **Push origin**.
5. Follow steps 10–13 above.
