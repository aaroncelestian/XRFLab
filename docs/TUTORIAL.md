# XRFLab Tutorial

A practical walkthrough from instrument setup to results: **blanks → FWHM → standards → single spectrum → batch**.

Launch the app from the project root:

```bash
pip install -r requirements.txt
python -m utils.sample_data   # optional: synthetic steel/brass/mineral
python main.py
```

Main tabs: **Analysis** | **Batch Analysis** | **Calibration**  
Calibration sub-tabs: **FWHM** | **Tube Profiles** | **Standards**

Do calibration first when you can. Analysis and Batch reuse whatever FWHM and tube profiles you applied.

---

## What you will get (and what you will not)

| Deliverable | Where | Meaning |
|-------------|--------|---------|
| Fit + peak list | Analysis → Results | Peak areas, energies, χ² / R² |
| **Semi-Quant** | Analysis → Results | Relative intensities from peak areas, normalized to 100% |
| FP / standards intensities | Calibration → Standards | Instrument / fundamental-parameters style path |

**Semi-Quant is not weight percent from fundamental parameters.** Treat it as a quick relative ranking of labeled sample peaks. Tube lines are excluded.

---

## Part 1 — Blanks and tube profiles

### Why blanks?

A **blank** (empty beam / scatter-dominated spectrum, no sample of interest) captures the **tube anode lines** and **Compton** scatter at a given tube voltage. XRFLab stores relative intensities for those lines per kV mode (15 / 30 / 50). During fitting, those ratios act as soft priors and help flag when a “tube” line is actually contaminated by a sample peak.

Detector **FWHM** is one energy-dependent curve. **Tube shape** is per voltage.

### Steps

1. Open **Calibration → Tube Profiles**.
2. Set **Tube anode** (typically **Rh**), **Tube voltage** (**15**, **30**, or **50** kV), scatter angle (often **90°**), and Compton FWHM if needed (default **250 eV**).
3. Click **Load Blank Spectrum…** and choose your blank (`.txt`, `.csv`, `.mca`, `.msa`, `.emsa`).  
   If the file metadata includes excitation energy, the voltage may snap to the nearest mode.
4. Click **Measure Profile**. Ratios are stored for that kV and the library is auto-saved.
5. Repeat for each tube voltage you actually use in the lab.
6. Click **Apply to Analysis**.

Optional: **Save Library…** / **Load Library…** for backups or sharing between machines.

### Tips

- There is **no blank** in the shipped `sample_data/` folder. Use a real instrument blank, or skip this part and rely on built-in default ratios (fine for learning the UI; weaker for real tube overlaps).
- Below ~20.5 kV, Rh **K** lines are typically off; 15 kV profiles are often L-line dominated. 30 and 50 kV usually include K lines + Compton.
- Library location (auto): under the app data calibrations folder (`tube_profiles.json`).

---

## Part 2 — FWHM (detector resolution)

### Why FWHM first?

Peak fitting needs a good guess (and often a lock) for peak width vs energy:

\[
\mathrm{FWHM}(E) = \sqrt{\mathrm{FWHM}_0^2 + 2.355^2 \cdot \varepsilon \cdot E}
\]

Pure-element spectra give clean peaks across a useful energy range. Apply FWHM before you trust fitting widths or standards work.

### Steps

1. Open **Calibration → FWHM**.
2. Click **Browse…** and select a **folder** of pure-element spectra (not a single file).
3. Choose model **Detector** (recommended). Linear / Quadratic / etc. are available for comparison.
4. Optional: tag tube voltage (Mixed / 15 / 30 / 50) for bookkeeping — FWHM remains one curve.
5. Click **Run**. You need enough resolvable peaks (aim for several across ~1–16 keV).
6. Check R², RMSE, and the FWHM vs energy plot. Then click **Apply**.
7. Optionally **Save…** a JSON copy.

### Example folder (included)

Use:

`sample_data/data/`

Useful files:

| File | Role |
|------|------|
| `Mg.txt`, `Ti.txt`, `Fe.txt`, `Cu.txt`, `Zn.txt` | Clean K-line peaks |
| `Al.txt` | Al; many metal foils also show holder Al Kα |
| `cubic zirconia.txt` | Zr L + K (wide range; can be an outlier) |

Precomputed calibrations you can **Load…** instead of running:

- `sample_data/data/fwhm_calibration.json`
- `sample_data/peak_shape_calibration.json`

### Quality targets

- R² ≳ **0.95**
- FWHM₀ typically **~80–150 eV**
- ε typically **~2–5 eV/keV**
- Residuals roughly ≲ **5 eV**

If R² is poor, drop weak or overlapping peaks (cubic zirconia is a common troublemaker) and re-run. See [TROUBLESHOOTING_CALIBRATION.md](TROUBLESHOOTING_CALIBRATION.md).

After **Apply**, the Analysis **Elements** / Fitting status should show FWHM as active.

---

## Part 3 — Standards (known compositions)

### Purpose

Standards with **known concentrations** support instrument intensity calibration and FP/fisx-style work. This is separate from Analysis Semi-Quant.

### Steps (load and prepare)

1. Confirm FWHM status is applied (green / ready) on **Calibration → Standards**.
2. **Add Standard** — give a name (e.g. `NIST 2586`).
3. Attach one or more spot spectra for that standard.
4. Enter concentrations (CSV or manual). For shipped CSVs with mg/kg, values above 100 are treated as mg/kg and converted to wt% in the UI.
5. Optionally add replicate spots with **Add Spectra…**.
6. On the Calibration sub-panel, set background (AsLS is a common choice), **Preview Background**, then use **Run Intensity Calibration** when available.

### Example files

| Kind | Path |
|------|------|
| Spectra | `sample_data/data/NIST 2586.txt`, `NIST 2587.txt` |
| Spot folders | `sample_data/data/Spectrum value from standard/Nist 2586/`, `Nist2587/`, also Till / PACS / LKSD / STDS |
| Concentrations | `sample_data/data/NIST_SRM_2586_elements.csv`, `NIST_SRM_2587_elements.csv` |

### Current limitation

**Multi-standard optimization is not implemented yet.** The Standards UI can load spectra and concentration tables, and you can load a previously saved calibration JSON if you have one, but **Run Intensity Calibration** does not yet compute a new multi-standard result. Single-spectrum **Analysis Semi-Quant** and **Batch** still work without it.

When you only need relative intensities on unknowns, skip ahead to Part 4.

---

## Part 4 — Single spectrum analysis

Work in the **Analysis** tab. Left sub-tabs in order:

**Sample/Exp → Peak Find → Elements → Fitting → Results**

### Step A — Open and describe the sample

1. **File → Open Spectrum** (or **Open Spectrum** on the toolbar / Sample/Exp).
2. Try `sample_data/steel_sample.txt` (after `python -m utils.sample_data`), or a real file such as `sample_data/data/stainless steel.txt`.
3. On **Sample/Exp**, set name/type if useful, and match **Excitation**, **Current**, **Live Time**, detector, and geometry to the measurement (use file metadata when present).

### Step B — Peak find + auto-ID

1. Open **Peak Find**.
2. Leave **Auto-find unknown peaks** and **Auto-ID peaks** checked.
3. Tune prominence / min height / separation if needed.
4. Click **Find Peaks + Auto-ID**.

Peaks are marked on the spectrum, listed for editing, and matched to common XRF lines. Suggested elements are selected on the **Elements** tab (you switch there when ready to review).

5. Delete obvious false peaks from the Found Peaks list before fitting.

### Step C — Review elements

1. On **Elements**, uncheck false IDs and add any missing elements.
2. Click an element to overlay its emission lines on the spectrum if helpful.

### Step D — Fit

1. Open **Fitting**.
2. Typical starting point: peak shape **Voigt**; post-fit Smart ID on for overlap review.
3. Click **Fit Spectrum** (`Ctrl+F`).

Watch the main plot and residuals. Tube-overlap flags (if any) appear under Results.

### Step E — Semi-Quant and export

1. Open **Results**.
2. Review fit statistics and the peak list.
3. Click **Semi-Quant** for area-normalized relative intensities.
4. Click **Export Results** when you want a table out.

### Checklist if Semi-Quant is empty

- Did you run **Peak Find + Auto-ID** (or manually select elements) and then **Fit**?
- Are the peaks only tube lines? (Tube lines are excluded.)
- Did you delete all labeled sample peaks from the peak list?

---

## Part 5 — Batch processing

Batch uses the **same fitter and Semi-Quant rules** as Analysis. Configure Analysis once, then process many files.

### Steps

1. On **Analysis**, set elements, excitation, background, peak shape, and tube options the way you want the whole batch to run.
2. Open **Batch Analysis → Setup**.
3. Click **Refresh** so the summary shows Elements / Beam / Fit / Tube from the Analysis tab.
4. **Add…** individual spectra or **Folder…** (loads `*.txt`, `*.csv`, `*.mca` from a directory).
5. Click **Process All** (requires at least one file and at least one element).
6. On **Results**, review the summary table (success, R², χ²), click a row to inspect a fit, use Trends if useful, then export CSV/Excel.

### Example batch folders

- `sample_data/data/Spectrum value from standard/Till 1/`
- `sample_data/data/Spectrum value from standard/Nist 2586/`

Keep instrument settings consistent across the batch. If some files were collected at a different kV, either split batches or update Analysis settings and Refresh before processing each group.

---

## Suggested first session (45 minutes)

1. **Load FWHM** from `sample_data/data/fwhm_calibration.json` (Calibration → FWHM → **Load…** → confirm Apply).
2. Skip blanks if you have none; note that tube priors are defaults.
3. Open `sample_data/steel_sample.txt`.
4. **Peak Find → Find Peaks + Auto-ID** → review **Elements** → **Fitting → Fit Spectrum** → **Semi-Quant**.
5. Point Batch at a small folder of similar spectra → Refresh → Process All → export.

---

## Keyboard shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+O` | Open spectrum |
| `Ctrl+F` | Fit spectrum |
| `Ctrl+Q` | Semi-Quant (relative intensities) |

Tab-specific buttons (**Fit Spectrum**, **Semi-Quant**, **Export Results**) live on the Fitting and Results panels, not on the global toolbar.

---

## Related docs

- [QUICKSTART.md](QUICKSTART.md) — install and first open
- [FITTING_GUIDE.md](FITTING_GUIDE.md) — backgrounds and peak shapes
- [CALIBRATION_WORKFLOW.md](CALIBRATION_WORKFLOW.md) — pure-element FWHM details
- [FWHM_INTEGRATION_GUIDE.md](FWHM_INTEGRATION_GUIDE.md) — FWHM into the rest of the pipeline
- [TROUBLESHOOTING_CALIBRATION.md](TROUBLESHOOTING_CALIBRATION.md) — poor R² / outlier peaks
- [../README.md](../README.md) — feature overview and limitations
