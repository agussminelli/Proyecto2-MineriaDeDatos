# Individual Research — Geographical Generalization in Verbal Autopsy Classification

Individual contribution to the shared **PHMRC-VAI cause-of-death classification project**, focusing on geographical generalization and site-associated vocabulary.

## Research Questions

- **RQ1:** To what extent does geographical distribution shift affect classification performance?
- **RQ2:** Can suppressing site-associated vocabulary improve cross-site cause-of-death classification while preserving performance on familiar sites?

## Data and evaluation setup

- **Data:** 5,209 narratives; 14 cause-of-death (CoD) classes; 6 sites (AP, Bohol, Dar, Mexico, Pemba, UP); Adult, Child and Neonate modules.
- **Shared group split:** Train 3,646 (70%), Dev 1,042 (20%), Test 521 (10%).
- **Text input:** `open_response_clean`; **target:** `gs_text34`; **group variable:** `site`.
- **Primary metric:** Macro-F1, including a consistent label set (`zero_division=0`). Accuracy and per-site/per-class F1 are supplementary.
- **Unseen:** LOSO on Train + Dev: train on 5 sites, evaluate on the held-out sixth site.
- **Seen development:** original Train → Dev, used during method selection.
- **Final test:** refit on Train + Dev (4,688) → Test (521); **familiar sites**, not a seventh site.

## Models and selected settings

| Model | Representation | Classifier | Notes |
| --- | --- | --- | --- |
| Group baseline | TF-IDF → TruncatedSVD (300 components) | LinearSVC | Defined by `create_SVM_pipeline()` in `train_baseline.py` |
| **E0** | Word TF-IDF | Logistic Regression | Unmodified feature representation |
| **Final E4** | Word TF-IDF, selected columns downweighted | Logistic Regression | Top **2%** Site-MI features multiplied by **0.50** |

E0 and E4 both use **`C=1.0`**, balanced class weights, up to **5,000 TF-IDF features**, **1–3-grams**, and **no SVD**. Earlier baseline experiments included SVD, one-vs-rest LR, and GroupKFold-based `C` selection; the feature experiments keep `C` fixed for comparison.

**Final feature-weighting procedure:**

1. Fit TF-IDF **only on the training narratives**.
2. Convert the training TF-IDF matrix to binary feature presence/absence.
3. Compute mutual information (**MI**) between each feature and the **training-site labels**.
4. Select the highest-ranked 2% of TF-IDF columns (100 of 5,000 features when the vocabulary reaches its maximum size).
5. Multiply these columns by **0.50** in both training and evaluation matrices; leave all other features unchanged.
6. Fit LR on the modified training matrix and predict the evaluation narratives.

**No leakage:** calculate MI separately within each LOSO training fold; for final Test, calculate it from Train + Dev only. Site MI indicates geographical association, **not necessarily nonclinical vocabulary**.

## Experiment sequence

| ID / stage | What was investigated | Outcome / next decision |
| --- | --- | --- |
| Baseline exploration | SVM, LR, one-vs-rest LR; with/without SVD; regularization checks | Continue with LR without SVD for the feature experiments |
| **RQ1: LOSO** | All 14 classes, plus Adult-only checks | Performance differs substantially between sites; site and module/CoD distributions are not identical |
| **RQ1: controlled LOSO** | Five causes represented at every site; matched cause quotas and training/evaluation sizes; Seen vs. Unseen on the same target records; 30 sampling seeds | Quantify the benefit of having target-site examples in training |
| **E1** | Hand-selected documentation terms (V1/V2); site-prediction check | Filtering reduces site predictability but CoD gains are small or inconsistent |
| **E2** | Remove top 1%, 2% or 5% of features ranked by Site MI | Modest average improvement in exploratory 14-class LOSO |
| **E3a / E3b** | Combine Site MI with CoD MI; E3a prioritizes low CoD MI, E3b protects high-CoD-MI candidates | CoD-aware feature protection is possible but does not consistently beat Site-MI-only selection |
| **E5b control** | Remove a comparable number of randomly selected, approximately document-frequency-matched features | Random removal performs worse on average; matching becomes imperfect at higher budgets |
| **E4** | Compare removal with partial downweighting (`0`, `0.25`, `0.50`) across feature budgets | Choose **E2 ranking, top 2%, weight 0.50** for its balance of Seen/Unseen performance |
| Robustness | Leave-one-site-excluded summaries, per-class changes and paired prediction changes | Check whether gains depend on one site or a few causes |
| Controlled E0 vs. E4 | Reuse the five-class matched Seen/Unseen design with 30 seeds | Test stability when cause distributions and training sizes are controlled |
| Final Test | Frozen E0 and E4, plus group SVM, refitted on Train + Dev | Independent **familiar-site** comparison |

The six LOSO sites were examined during method development; these results are **exploratory**, not an independent external validation.

## Relevant code files

| File | Role |
| --- | --- |
| `logistic_regression.py` | Shared data loading/preprocessing (`load_and_preprocess`) and LR pipeline (`create_LR_pipeline`) |
| `loso_sets.py` | Ordinary and balanced controlled LOSO split generators |
| `run_experiment.py` | **Main experiment runner** (previously named `loso_experiments.py`). Runs the individual RQ1/RQ2 experiments selected through Hydra configuration, including LOSO, controlled LOSO and E0–E5. Also provides the reusable `apply_site_mi_weighting()` helper. |
| `data_exploration.py` | Three separate data-exploration functions. Select the one(s) to run by commenting/uncommenting function calls in its `if __name__ == "__main__":` section. |
| `result_examination.py` | Three separate result-examination functions; selected manually in its `if __name__ == "__main__":` section in the same way. |
| `site_vocabulary_audit.py` | Exploratory inspection of site-associated terms and documentation styles |
| `robustness_analysis.py` | Post-hoc E4 comparison using saved result/prediction CSVs; no model fitting |
| `train_baseline.py` | Group baseline: TF-IDF → SVD → LinearSVC |
| `final_seen_test.py` | Final Train + Dev → Test evaluation of the group SVM baseline, E0 and E4, including bootstrap confidence intervals. Calls `apply_site_mi_weighting()` from `run_experiment.py`. |
| `conf/` | Hydra configuration (data columns, vectorizer, classifier, validation and controlled-LOSO settings) |



## Running the code

Run the scripts from the **repository root**. Configure the dataset paths and experiment settings in `conf/` first. The code uses `numpy`, `pandas`, `scipy`, `scikit-learn`, `hydra-core`, `omegaconf` and `joblib` (plus any other packages specified by the group environment).

### Main experiment runner: `run_experiment.py`

All individual research experiments are launched from **`run_experiment.py`**. The experiment to execute is selected by the Hydra configuration (`cfg.experiment.name`); there is no separate Python file for each E0–E5 experiment.

```bash
# Run the experiment selected in conf/config.yaml
python run_experiment.py

# Example: initial LOSO model comparison
python run_experiment.py experiment.name=loso_LR

# Example: original controlled LOSO
python run_experiment.py experiment.name=loso_LR_controlled
```

The available experiment names are taken from `main(cfg)` in `run_experiment.py`:

| Hydra `experiment.name` | Function / purpose |
| --- | --- |
| `loso_LR` | `run_loso_experiments`: initial LOSO model comparison, including SVM and LR variants and grouped C selection |
| `loso_LR_controlled` | `run_controlled_loso_experiment`: controlled five-class Seen/Unseen LR baseline |
| `loso_LR_controlled_E4` | `run_controlled_loso_experiment(..., apply_site_weighting=True)`: paired E0/E4 controlled LOSO |
| `site_prediction` | `run_predict_site_exp`: predict geographical site with/without the manual vocabulary filter |
| `module_influence_on_site` | `run_module_only_site_pred`: predict site from module alone |
| `E2_autocut_top_site_features` | `E2_autocut_top_site_features`: remove top Site-MI features |
| `E3_autocut_site_cod_features` | `E3_autocut_site_cod_features`: CoD-aware feature selection; variant controlled by `feature_filter.ordering_method` |
| `E5_frequency_matched_random` | `E5_frequency_matched_random`: frequency-matched random removal control |
| `E_seen_feature_comparison` | `E_seen_feature_comparison`: Train → Dev comparison of E0, E2, E3a and E3b |
| `E4_feature_downweighting` | `E4_feature_downweighting`: compare removal and partial downweighting for Seen and Unseen |

```bash
# Automatic Site-MI feature removal (E2)
python run_experiment.py experiment.name=E2_autocut_top_site_features

# CoD-aware alternative (E3a or E3b)
python run_experiment.py experiment.name=E3_autocut_site_cod_features feature_filter.ordering_method=cod_first
python run_experiment.py experiment.name=E3_autocut_site_cod_features feature_filter.ordering_method=site_first_protected

# Random-control experiment (E5b)
python run_experiment.py experiment.name=E5_frequency_matched_random

# Removal vs. downweighting comparison (E4)
python run_experiment.py experiment.name=E4_feature_downweighting

# Controlled Seen/Unseen E0 vs. E4 comparison
python run_experiment.py experiment.name=loso_LR_controlled_E4

# Seen Train -> Dev feature comparison
python run_experiment.py experiment.name=E_seen_feature_comparison
```

The E5b experiment reads the E2 feature file specified by `feature_filter.e2_removed_features_path`; check that path and `feature_filter.random_repeats` in Hydra before running it. For E3, the two `ordering_method` values shown above correspond to the actual branches in the code. The manual E1 vocabulary experiments are implemented through the `filter_voc` setting in existing experiments rather than through a separate E1 dispatcher entry.

Hydra writes results into the configured output directory, usually in a run-specific folder. Preserve the configuration and outputs for reproducibility.

**Before submitting the code:** The uploaded development version of `run_experiment.py` still contains a temporary **resume mechanism** in `run_controlled_loso_experiment()` and a hard-coded output path (`output/loso_LR_controlled_E4/2026-11-23_20-25-40/`). These were added only to continue a long controlled experiment. Revert these changes for the submitted version, restoring `Path(HydraConfig.get().runtime.output_dir)` and a fresh empty `results` list, and remove the completed-run skip logic. Otherwise a new run may silently reuse an old CSV instead of starting a clean experiment. The 30-seed results already generated can still be retained as output artifacts.

### Data exploration and result examination

These two scripts are **not selected through Hydra**. Each contains **three functions** whose calls are manually enabled or disabled near the bottom of the file, under `if __name__ == "__main__":`.

```bash
python data_exploration.py
python result_examination.py
```

Before running either script, open it and **uncomment the function call(s) you want to execute; comment out the others**. Different functions may produce different tables, plots or saved outputs. The exact function names and their individual output filenames are documented in the source code.

### Final Seen-site test

The final model comparison is run separately, after the exploratory experiments and feature-weighting configuration have been fixed:

```bash
python final_seen_test.py
```

This script retrains the **group SVM**, **unweighted LR (E0)** and **weighted LR (E4)** on Train + Dev and evaluates all three on the same untouched Test partition. It reports overall/per-site/per-class performance and 95% site-stratified bootstrap intervals (1,000 repetitions) for model scores and paired differences. The final Site-MI feature weighting is imported from `run_experiment.py`.

### Additional analysis scripts

```bash
# Optional exploratory vocabulary audit
python site_vocabulary_audit.py

# Summarize saved E4 experiment results without retraining
python robustness_analysis.py
```

For `robustness_analysis.py`, set `E4_DIR` to the relevant Hydra output directory containing `e4_results.csv` (and `e4_predictions.csv` for per-class analyses).

## Main output files

| File(s) | Content |
| --- | --- |
| `loso_all.csv`, `loso_summary.csv`, `predictions.csv` | Initial LOSO fold metrics, overall summaries and predictions |
| `controlled_site_results.csv`, `controlled_site_summary.csv`, `controlled_site_deltas.csv`, `controlled_site_gap_comparison.csv` | Seed × site × condition controlled results and E4–E0 / gap comparisons (from the extended controlled experiment) |
| `e4_results.csv`, `e4_predictions.csv`, `e4_selected_features.csv`, `e4_summary.csv` | Downweighting results, predictions, selected features and summaries |
| `robustness_overview.csv`, `robustness_per_class.csv` | Site sensitivity and class-level robustness summaries |
| `final_test_results.csv`, `final_test_comparison.csv`, `final_test_mean_sites.csv` | Final SVM/E0/E4 overall and site-level performance |
| `final_test_predictions.csv`, `final_test_per_class.csv` | Individual test predictions and performance by cause (including SVM if using the updated final test script) |
| `final_test_bootstrap.csv`, `final_test_selected_features.csv` | Percentile bootstrap intervals and selected Site-MI features |

## Selected results (reference)

| Evaluation | E0 Macro-F1 | Weighted LR Macro-F1 | Difference |
| --- | ---: | ---: | ---: |
| Exploratory 14-class LOSO, **mean across sites** | 0.2941 | 0.3044 | +0.0103 |
| Controlled 5-class LOSO, **Unseen** (30 seeds) | 0.5000 | 0.5173 | +0.0173 |
| Controlled 5-class LOSO, **Seen** (30 seeds) | 0.6048 | 0.6124 | +0.0076 |
| Final 14-class **Test** (521 records) | 0.4677 | 0.5224 | +0.0546 |

On the final Test set, the group SVM obtained **Macro-F1 0.4460**; E4 improved accuracy from **58.35% to 60.46%** versus E0. The paired, site-stratified bootstrap (1,000 resamples) gave a **95% interval of [+0.0021, +0.1067]** for the final E4–E0 Macro-F1 difference.

### Interpretation and limitations

- Controlled LOSO's **five-class** Macro-F1 is not numerically interchangeable with the **14-class** LOSO or Test metrics.
- The final Test is independently held out but contains the **same six geographical sites** represented in training. It validates familiar-site performance, **not** transfer to a new seventh site.
- The controlled 30 seeds reuse overlapping observations; they measure sensitivity to sampling, **not 30 independent datasets**.
- Site-MI downweighting can also affect clinically useful terms. Observed improvements do not prove that only documentation artifacts were removed.
- Rare classes matter: a two-case nervous-system category contributes heavily to final Macro-F1. The mean difference over the remaining 13 class-F1 scores is **+0.0204**.
- The E0–E4 comparison isolates the representation change with the same LR settings. SVM-versus-E4 compares **complete pipelines** (different representation and classifier).

Detailed decisions, figures and paper notes are in the **Individual Research Diary**.
