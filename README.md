# FI-Lab in older ICU patients: derivation and multi-database external validation

Analysis code for a study of a **laboratory-based frailty index (FI-Lab)** in critically ill
adults aged 65 years or older. FI-Lab is a 30-item Rockwood deficit-accumulation index built
only from laboratory tests drawn on the first ICU day. The index is derived in MIMIC-IV and
then transported with **frozen coefficients** to two independent databases (NWICU and eICU-CRD).

The study asks two separate questions, and answers them separately:

1. Does FI-Lab carry prognostic information beyond age, comorbidity and acute severity?
2. Does a model containing FI-Lab keep its **discrimination and its absolute risk** when it is
   applied unchanged in another hospital's database?

Discrimination transported moderately well (AUC 0.714 in NWICU, 0.736 in eICU). Calibration
did not transport unmodified (observed-to-expected ratio 1.17 and calibration-in-the-large
0.20 in NWICU). We report both results as they are; the code here reproduces every number,
table and figure in the manuscript.

## Repository layout

```
code/          analysis pipeline (Python 3, PostgreSQL)
results/       aggregate result tables (no patient-level rows)
figures/       final manuscript figures
```

### code/ — pipeline order

The scripts are numbered by the order in which they were run. Each script is standalone and
reads the database directly; nothing is cached between steps except the CSV/JSON files written
to `analysis/`, `cohort/` and `filab/` (all gitignored).

| Stage | Scripts | What they do |
|---|---|---|
| 1. Cohort | `build_cohort.py` | Adults ≥65, first ICU stay within each hospitalization, per database |
| 2. FI-Lab | `build_filab.py` | 30-item deficit count, per-item normal ranges, ≥15/30 assessable rule |
| 3. Outcomes | `build_outcomes.py` | In-hospital death, 30-day death, 30-day readmission, ICU/hospital LOS |
| 4. Severity | `_augment_severity.py` | SOFA, OASIS, ventilation/vasoactive/RRT flags, age-free Charlson (MIMIC-IV derived table; ICD-9/ICD-10 + Quan mapping for NWICU) |
| 5. Models | `build_analysis.py` | M1–M4 sequential adjustment, frozen-coefficient external validation |
| 6. Sensitivity | `build_sensitivity.py`, `sensitivity_lowmissing_*.py` | COVID-era, ARD subtype, low-missingness panels |
| 7. Revision analyses | `revision_analysis.py`, `_revision_ext.py`, `_revision_tables.py`, `table2_auc_ci.py`, `_rev5_tables.py`, `_extended_30d.py` | Calibration (slope, CITL, O:E with Poisson CI), bootstrap AUC, DCA, cluster-robust SE |
| 8. Database audits | `_db_*.py`, `_rev7_*.py`, `_rev8_s6b.py`, `_frozen_coef_table.py` | ARD code contamination check, NWICU death-date audit, 14-item harmonized panel, early-death sensitivity, per-laboratory missingness |
| 9. ARD sensitivity (superseded) | `_recompute_nonard*.py`, `_recompute_rev2_*.py` | Full recomputation on an ARD-excluded cohort. Kept for transparency; **not** the analysis reported in the manuscript |

Scripts whose names start with `_` are the revision-stage analyses, run in the order of the
manuscript's revision history. The final numbers come from stage 7–8.

### results/ — aggregate tables

| File | Manuscript location |
|---|---|
| `baseline_table1.csv` | Table 1 (baseline characteristics) |
| `hierarchical_models_mimiciv.csv`, `incremental_value_mimiciv.csv`, `rev2_increment_ci.csv` | Table 2 / Figure 2 (M1–M4, incremental value) |
| `external_validation_frozen.csv`, `frozen_coefficients_full.csv`, `calibration_points.csv` | Table 3 / Figure 3 (frozen-coefficient external validation) |
| `dca_mimiciv.csv`, `dca_plot_mimiciv.csv`, `dca_mimic.csv` | Figure S2 (decision curve) |
| `per_lab_missingness.csv`, `db_audit/per_lab_missingness_final.csv` | Figure S1 (missingness heatmap) |
| `sens_lowmissing_*.csv` | Supplementary tables S6–S7 (restricted and 14-item panels) |
| `ard_subgroup_performance.csv`, `rev2_ard_analysis.csv`, `ard_icd_codes.csv` | Supplementary ARD analyses (code contamination audit) |
| `attrition_counts.csv` | Figure 1 (flow diagram) |
| `_root/*` (`results/`) | 30-day and readmission outcomes, era/COVID/subtype sensitivity |

Every file in `results/` is a decile-, threshold- or model-level summary. No row corresponds
to a patient.

## Data access

None of the source data can be redistributed here. All three databases are available from
PhysioNet to credentialed users who complete the required training and sign a data use
agreement:

- **MIMIC-IV** v3.1 — https://physionet.org/content/mimiciv/
- **eICU-CRD** v2.0 — https://physionet.org/content/eicu-crd/
- **NWICU** — https://physionet.org/content/nwicu/

The scripts expect these databases to be installed in a local PostgreSQL instance, using the
schema names of the PhysioNet distributions: `mimiciv_hosp`, `mimiciv_icu`, `mimiciv_derived`
and `eicu_crd`. NWICU tables are referenced as `nwicu.*`, `icu.*` and `hosp.*`.

## Running the code

```bash
python -m pip install -r requirements.txt

export FILAB_DB_HOST=localhost      # default 127.0.0.1export FILAB_DB_PORT=5432
export FILAB_DB_USER=postgres
export FILAB_DB_PASSWORD=           # no default, must be set

cd code
python build_cohort.py              # writes ../cohort/
python build_filab.py               # writes ../filab/
python build_outcomes.py
python _augment_severity.py
python build_analysis.py            # writes ../analysis/
```

Output directories (`cohort/`, `filab/`, `analysis/`, `figures/`) are created on import and
are gitignored.

Counts in this repository are those reported in the manuscript (45,656 MIMIC-IV, 10,819 NWICU
and 56,544 eICU hospitalizations with an assessable FI-Lab). They depend on the database
snapshot and on the `hadm_id`/`stay_id` content of that snapshot, so small differences are
expected if a different extract is used.

### code/FI_Lab_Validation_Template.csv

A blank data-entry template for groups who want to validate the frozen model in their own
hospital, used by `compute_FI_from_template.py` (Rockwood deficit count + frozen M2 model →
linear predictor, predicted probability, AUC, calibration). The file contains **two synthetic
example rows** (`EX001`, `EX002`) that exist only so the script runs out of the box and so users
can check their installation. They are invented, are not derived from any patient record, and
must be deleted before entering real data.

## Deliberately not included

- **Patient-level data.** Every `.csv` produced by the pipeline is per-hospitalization and is
  excluded. Only identifier-free aggregates are published, in `results/`.
- **Manuscript and supplementary builders.** The scripts that assemble the Word documents,
  declare author contributions, and generate the ethics application are not reproducibility
  artefacts, and they embed unpublished text. They are held by the authors.
- **Exploratory item-mapping scratch scripts.** They only document how laboratory `itemid`s
  were identified and are superseded by `build_filab.py`.

Comments in the code are in Chinese. They record variable definitions, cohort logic and
missingness warnings and are part of the reproducibility record; only tooling artefacts were
removed.

## Environment

Python 3.10+, PostgreSQL 14+. Tested with psycopg2 2.9, pandas 2.x, numpy 1.26,
statsmodels 0.14, scikit-learn 1.4, scipy 1.11, matplotlib 3.8.

## Citation

If you use this code, please cite the article (DOI to be added on publication) and the archived
release of this repository. The concept DOI resolves to the latest version; each release has
its own version DOI.

## License

MIT — see `LICENSE`. The license covers the code in this repository only, not the source
databases, which remain governed by their PhysioNet data use agreements.
