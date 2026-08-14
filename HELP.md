# ActMOF — User Guide

This guide explains how to use the ActMOF web app to plan and run a Bayesian-optimization
(BO) synthesis campaign for Metal-Organic Frameworks (MOFs). Every section pairs a plain-language
explanation with the precise, rigorous definition underneath it, so you can skim for "what do I click"
or dig into "what is the app actually computing."

The same content (condensed) is available inside the app itself, under the **❓ Help & Guide** tab.

---

## 1. What this app does

**In plain terms:** You tell the app 5 knobs you can turn when making a MOF (metal amount, modulator
amount, extra solvent, reaction time, reaction temperature). The app suggests a small batch of
knob-settings to try next. You run those reactions, measure the XRD pattern, type in the results, and
the app learns from them and suggests the next, smarter batch. Repeat until you've found conditions
that reliably make sharp, well-crystallized MOF.

**Rigorously:** This is sequential, batched Bayesian optimization over a 5-dimensional discrete design
space. A Gaussian Process (GP) surrogate model is fit to your accumulated `(condition → q)`
observations; an acquisition function (Expected Improvement or Probability of Improvement) scores
unseen candidate conditions; a diversity-aware greedy selection picks a batch of `k` conditions that
are jointly high-acquisition and spread out in feature space, to avoid wasting a batch on near-duplicate
experiments.

---

## 2. The five synthesis parameters

| Parameter | Meaning | Range |
|---|---|---|
| `metal_amount` | Amount of metal precursor | 5–75 |
| `modulator` | Amount of modulator (controls crystal growth/defects) | 5–15 |
| `add_solvent` | Additional solvent volume | 0–30 |
| `reaction_time` | Reaction duration | 1–12 |
| `reaction_temperature` | Reaction temperature | 10–30 |

These bounds are fixed in the app (`bo_engine.py`, `BOUNDS`); the app will never suggest a value
outside them, and integer step size is 1 for every parameter.

---

## 3. The `q` score — your optimization target

**Plain language:** After you run a reaction, you scan the resulting powder with XRD. If the MOF
crystallized well, you get one tall, narrow peak. `q` is just a single number that rewards "tall and
narrow": bigger q = sharper, more crystalline MOF. A failed/amorphous reaction shows a flat, noisy
pattern with no real peak, and gets `q = 0`.

**Rigorously:** `q = peak_intensity / FWHM`, where `FWHM` (full width at half maximum) is the width of
the dominant diffraction peak measured at half its height, found by linear interpolation between the
two points nearest the peak on the intensity trace. This is computed automatically when you upload a
`.rasx` file, or by typing `intensity` and `fwhm` directly (`calc_q()` in `bo_engine.py`). If you already
know `q`, you can enter it directly instead. The app is maximizing q — high peak intensity and a narrow
peak width, i.e. it is chasing crystallinity/phase purity, not yield or particle size directly.

---

## 4. Rounds, batches, and status

Every proposed condition becomes one row in your experiment log. A **batch** is a set of `k`
conditions proposed together (`k` = "Experiments per Batch" in the sidebar); a **round** is the
0-indexed counter for batches (`round = batch_number - 1`). Each row also has a `status`:

- `suggested` — proposed by the BO engine, no measurement entered yet.
- `completed` — you entered a valid measurement (or `q`), so it now counts as training data.
- `cancelled` — you can mark a row this way (e.g. in the CSV) to exclude it from the model without
  deleting it.

Record IDs are formatted `R00001`, `R00002`, … in the order rows were created, matching the format of
an exported `experiments.csv`.

---

## 5. Step-by-step workflow

1. **(Optional) Planning Wizard tab** — enter how many experiments per batch you can realistically run,
   and your best guess at how many batches the whole project will take (a min–max range). Click
   *Apply Wizard Settings* to save these into your project config. This range is only used to size the
   calibrated-transfer window (see §7) — it does not limit how long you can keep optimizing.
2. **Sidebar → Optimization Setup** — set your project name, batch size, GP kernel, and acquisition
   function. Defaults (Matérn 5/2 kernel, Expected Improvement) are good for most users; you generally
   only need to touch these if you have a strong reason to.
3. **(Optional) Enable Calibrated Transfer Prior** — turn this on if you want the very first few batches
   to be informed by the built-in benchmark dataset rather than pure random exploration. See §7.
4. **Batch Recommendations tab → Initialize BO Optimization** — generates your first batch of
   conditions. This button is disabled once a batch is active in the current session; it re-enables
   after you save results or upload a fresh CSV.
5. **Run the reactions**, then either:
   - Type `intensity` + `fwhm` directly into the results table, or
   - Upload the `.rasx` XRD file for a row and click *Fill Selected Row with Calculated Measurements*
     — the app computes intensity, FWHM, and q for you and shows the fitted half-width markers on the
     diffractogram.
6. Click **🔄 Update Model & Save Batch Results** — this saves your measurements into the experiment
   log, immediately re-fits the model, and proposes the next batch. Repeat from step 5.
7. **Dashboard tab** — at any point, check the 6-panel diagnostics: best-q-so-far progress, q per
   experiment, observed-vs-predicted parity (model fit quality), predicted-mean-vs-uncertainty
   landscape, how the acquisition function is ranking candidates, and a 2D map of where you've sampled.
8. **Experiment Data & History tab** — upload a prior `experiments.csv` to resume a project, edit any
   row by hand, or export your full history at any time.

---

## 6. How the recommendation engine decides

**Plain language:** Early on, with fewer than 3 completed results (and transfer prior off), the app
just samples spread-out random conditions — there isn't enough data yet to learn a trend. Once you have
3+ results with some variation in q, it fits a model of "what q would we expect at condition X, and how
sure are we?" and proposes the batch of conditions that best balances *predicted high q* against
*genuine uncertainty* (so it doesn't get stuck exploiting one lucky point), while keeping the batch
spread out so you're not wasting reactions on near-duplicates.

**Rigorously:** A NumPy-only Gaussian Process (Matérn 3/2 or 5/2 kernel, your choice) is fit to
`(scaled condition) → target`, where target is optionally `ln(1 + q)` (default on, reduces the effect
of outlier-large q values). Kernel length-scale and observation noise are chosen by grid search over a
fixed candidate set, minimizing the negative log marginal likelihood, with Cholesky-jitter fallback for
numerical stability. Roughly 15,000 random candidate conditions are scored with the chosen acquisition
function — Expected Improvement or Probability of Improvement, both computed against the best
completed-and-model-scaled result — and a batch of `k` is picked by greedy selection that trades off
acquisition score against minimum scaled-Euclidean distance to already-chosen points in the batch
(`diversity_lambda` controls how strongly diversity is weighted).

---

## 7. Calibrated Transfer Prior ("Mode C")

**Plain language:** If your project is brand new, the model has *nothing* to learn from for its first
few batches, so its early suggestions are essentially random guesses. This app ships with a built-in
reference dataset of prior MOF syntheses. Turning on "Calibrated Transfer Prior" lets the app borrow
patterns from that reference data for your first `M` batches, so your early recommendations are
smarter than pure guessing — and as your own results come in, the app blends in more and more of your
own data and relies less on the reference set.

**Rigorously:** A reference GP is fit once on the built-in benchmark set (96 conditions with known q).
For each of your own completed points, its reference-GP prediction `mu_ref(x)` is linearly calibrated
against your actual `y = ln(1+q)` via ridge-regularized least squares, `y ≈ a + b·mu_ref(x)`, with the
ridge term pulling `b` toward 1 when you have very little data of your own (regularization strength
`1/n_student`, `b` clipped to `[-2, 3]`). With zero student points, `a=0, b=1` (pure reference signal).
Once you have 3+ points, a second lightweight GP is fit on the *residuals* between your actual results
and the calibrated reference prediction, so any systematic quirks of your specific setup get modeled
too. The transfer prior is only used for the first `M` batches, where
`M = floor(mean(planned_iteration_min, planned_iteration_max) × transfer_fraction)` (or set manually).
After batch `M`, the app automatically switches to a purely student-data GP as described in §6.

---

## 8. Reading the Dashboard

| Panel | What it tells you |
|---|---|
| Best q so far | Running best q across completed experiments — should trend up. |
| q by completed experiment | Raw q per experiment, in run order — useful for spotting noisy batches. |
| Observed vs predicted q | How well the model's predictions match reality; points near the diagonal = good fit. |
| Uncertainty landscape | Where the model is confident (low spread) vs unsure (high spread) among candidates. |
| How suggestions are proposed | Acquisition score vs predicted q for all candidates — stars mark your currently open suggestions. |
| Experiment map | `metal_amount` vs `reaction_temperature`, colored by q, so you can see the region you've explored. |

---

## 9. CSV import/export format

Uploaded CSVs must contain the five feature columns (`metal_amount`, `modulator`, `add_solvent`,
`reaction_time`, `reaction_temperature`). If `status` is missing it defaults to `completed`; if `q` is
missing but `intensity`/`fwhm` are present, `q` is computed for you. Exported files always use the full
schema (`record_id, round, batch_position, status, <features>, intensity, fwhm, q, predicted_q_mean,
predicted_q_sd, acquisition_value, notes, created_at, updated_at`) — see `experiments.csv` in this repo
for a worked example spanning an initial calibrated-transfer batch through several rounds of
student-only optimization.

---

## 10. Troubleshooting / FAQ

- **"Initialize BO Optimization" is greyed out.** It's disabled once you already have an active batch
  of suggestions in this browser session. Save results with *Update Model*, or upload a new CSV
  (which clears the active batch), to re-enable it.
- **Dashboard says "No completed experiments yet."** The Dashboard only reflects *completed* rows.
  A freshly initialized batch is `suggested`, not `completed`, until you enter results and click
  *Update Model & Save Batch Results*.
- **My predicted q values look huge / much bigger than any q I've measured.** This is most visible in
  the reference-only phase of the calibrated transfer prior (`n_student=0`), where the model is
  extrapolating purely from the built-in benchmark GP. It will pull toward realistic values as soon as
  a few of your own completed results are available to calibrate against.
- **The model status says "Initial sampling was used…" or "Exploratory sampling…".** This means either
  you have under 3 completed current-project points and transfer prior is off, or your q values so far
  are all identical — the app is deliberately exploring rather than exploiting until there's a real
  trend to model.
- **I want to reset everything.** Sidebar → *Clear All Data*. This clears the experiment log and any
  active suggestions in this session (it does not touch anything you've already exported).
