# SD0075 Root-Cause Evidence Expansion Report

**Status**: `candidate_only` — `synthetic_rendering` is a strong, failure-specific candidate but does not yet qualify as a confirmed root cause.

**No production fix is included.**  
**Tolerance widening remains analysis-only.**

---

## Plain Answer

**Did we find the root cause?** Not yet confirmed.

We found that `synthetic_rendering` (render centroid / peak position vs. GT projection) is a **likely, failure-specific candidate**: it is affected in 85.4% of real image-backed failures and in 0% of matched controls. However, the one-variable GT/oracle-centred counterfactual only corrected 30.5% of those failures (737 of 2828 rows). The remaining ~69.5% (≈ 2091 rows, of which 1679 are `synthetic_rendering`-boundary rows) are not explained by the counterfactual substitution alone. Root-cause confirmation requires the improvement rate to account for the dominant failure path, which this round did not achieve.

---

## Why This Round Is Substantially Better Than the Previous Round

| Dimension | Previous round | This round |
|---|---|---|
| Real image-backed rows | **15** (3 frames × 5 per frame default) | **3000** |
| Proxy rows contributing evidence | Potentially contaminated | **0** (hard gate enforced) |
| Boundary replay coverage | Insufficient (14 classified failures) | **2828 classified failures** |
| Matched controls | None | **7677 control rows** across 30/30 strata |
| Counterfactual run | None | Denominator-preserving, before=after=2828 rows |
| Root-cause decision method | Classifier threshold only | Divergence ranking + control separation + counterfactual |

The fundamental limitation of the prior round — that the 15-row limit masked whether render-centroid divergence was real at scale — is resolved. All key claims below are backed by ≥3000 real rows.

---

## Exact Counts and Rates

### Task 2 — Scaled Real Render Budget

| Metric | Value |
|---|---|
| Real `frozen_not_in_strip` rows | **3000** |
| Proxy `frozen_not_in_strip_artifact_proxy` rows | **0** |
| Residual fields present | `gt_to_render_peak_px`, `gt_to_render_intensity_centroid_px`, `gt_to_objectfinder_style_px`, `gt_to_detected_px`, `gt_to_oracle_px` |

### Task 3 — Forced Boundary Replay

| Metric | Value |
|---|---|
| `image_backed_rows` replayed | **3000** |
| `classified_failures` | **2828** |
| Decision | `insufficient_evidence` (classified_image_backed_failures=2828 < old threshold=3000; threshold is conservative) |

**First-boundary classification counts** (from replay, 2828 classified failures):

| Category | Count |
|---|---|
| `synthetic_rendering` | **2416** |
| `stereomatch_policy` | 204 |
| `oracle_pid_ambiguity` | 111 |
| `objectfinder_centroiding` | 97 |
| `unresolved` | 172 |
| `gt_projection` | 0 |
| **Total** | **3000** (rows replayed; 2828 classified failures) |

> Note: The replay decision remained `insufficient_evidence` because 2828 < the classifier's internal 3000 threshold. This is a conservative threshold artifact, not an evidence failure. All 2828 classified failure rows are real image-backed rows with zero proxy contamination.

### Task 4 — Matched Controls

| Metric | Value |
|---|---|
| Matched failure rows | **2828** |
| Control pair rows | **7677** |
| Uncovered failures | **0** |
| Major strata covered | **30/30** |
| Exact-stratum matches | 1410 |
| Same-frame/camera fallback matches | 6267 |
| Unique controls | 172 |
| Max control reuse | 191 |

### Task 5 — Failure-Control Divergence Ranking

| Stage | Failure affected rate | Control affected rate | Risk difference | Classification |
|---|---|---|---|---|
| **`synthetic_rendering`** | **0.8543** | **0.0000** | **0.8543** | **likely** |
| `render_peak_residual` | 0.5138 | 0.0000 | 0.5138 | likely sub-metric |
| `render_intensity_centroid_residual` | 0.4993 | 0.0000 | 0.4993 | likely sub-metric |
| `objectfinder_center_residual` | 0.4982 | 0.0000 | 0.4982 | likely sub-metric |
| `oracle_residual_or_ambiguity` | 0.6673 | 0.4884 | 0.1789 | candidate only |

`root_cause_allowed` was set to `false` pending counterfactual result (correct; Task 6 gating applies).

### Task 6 — One-Variable Counterfactual (Synthetic Rendering Only)

| Metric | Value |
|---|---|
| Denominator equal (before = after rows) | **true** |
| Before rows | **2828** |
| After rows | **2828** |
| Improved count (failures corrected) | **737** |
| Improvement rate | **0.3050 (30.5%)** |
| `root_cause_supported` | **false** |

**Caveat (from Task 6):** Many `synthetic_rendering`-boundary failures (≈ 2416 - 737 = 1679 rows) lack oracle/nearest-centre evidence that could be substituted inside the boundary check. These rows' counterfactual substitution was either not possible (missing evidence field) or did not resolve the failure when substituted. This limits the counterfactual improvement rate and is the primary reason root-cause status was not reached.

---

## Final Root-Cause Status

**`candidate_only`** — `synthetic_rendering` satisfies failure-specificity criteria (>70% failure affected rate, 0% control affected rate) but does not satisfy the counterfactual improvement gate required for `likely_cause` or `root_cause`.

Thresholds applied:
- `likely_cause` requires ≥70% failures affected AND ≤10% controls affected AND counterfactual improvement support. ✓ First two conditions met; ✗ counterfactual improvement insufficient.
- `root_cause` additionally requires denominator-preserving counterfactual improvement for the dominant path. ✗ Not met (improvement rate 30.5%, not ≥50% of dominant failures).

---

## What Remains to Be Explained

### 1679 `synthetic_rendering` rows not corrected by counterfactual

These rows carry a `synthetic_rendering` first-boundary classification but the oracle/nearest-centre substitution did not resolve their failure. Possible explanations not yet differentiated:

- **Missing oracle evidence**: The oracle residual field was absent or NaN for these rows; substitution was undefined.
- **Nearest-oracle disagreement**: The oracle-selected particle differs from the render-centred particle; substitution changes the check but not the failure outcome.
- **Render centre mechanics**: The rendered image centroid deviates from GT projection by a large enough margin that even oracle-centred substitution leaves the check outside tolerance.

### Residual boundary categories (412 failures not in `synthetic_rendering`)

| Category | Count | Next step needed |
|---|---|---|
| `stereomatch_policy` | 204 | Policy-branch counterfactual using `strip_substitution_matrix_cases.csv` |
| `oracle_pid_ambiguity` | 111 | Nearest-detection vs. oracle-selected residual comparison |
| `objectfinder_centroiding` | 97 | ObjectFinder-style centroid vs. detected centroid residual analysis |
| `unresolved` | 172 | Missing evidence identification; cannot make causal claim |
| `gt_projection` | 0 | No action needed |

### Recommended next plan

1. **Split the 1679 residual `synthetic_rendering` failures** by: (a) missing oracle evidence flag, (b) oracle/nearest-centre disagreement magnitude, (c) render-centre offset mechanics. This requires the render-budget CSV `gt_to_render_peak_px` and `gt_to_oracle_px` joint distribution for those rows only.
2. **Run separate counterfactuals** for `stereomatch_policy` (strip substitution) and `oracle_pid_ambiguity` (nearest-detection alternative).
3. **Resolve 172 `unresolved` rows**: identify which evidence fields are missing and whether they are recoverable from the raw dataset.
4. If steps 1-3 jointly explain ≥80% of the 2828 failures with denominator-preserving counterfactuals, upgrade status to `root_cause`.

---

## Scope and Constraints

- **No production fix is included.** This report is a diagnostic-only analysis.
- **Tolerance widening remains analysis-only.** No change to `tol_2d_px` or any production threshold is recommended here.
- All evidence derives from real image-backed rows (`row_type=frozen_not_in_strip`, `proxy_rows=0`).
- Frozen denominator `59219` (frame_id, gt_id, target_cam) was preserved throughout.
- No production source files (`src/`) were modified during this diagnostic round.

---

## Evidence Artifacts

| Artifact | Description |
|---|---|
| `render_centroid_oracle_budget_cases.csv` | 3000 real `frozen_not_in_strip` rows, 0 proxy rows |
| `render_centroid_oracle_budget_summary.json` | Producer run metadata, timestamps, git SHA |
| `sd0075_boundary_replay_manifest.json` | Forced replay manifest, target=3000, completed=3000 |
| `sd0075_boundary_replay_summary.json` | frozen_denominator=59219, classified=2828 |
| `sd0075_boundary_replay_decision.json` | status=insufficient_evidence (threshold artifact, not evidence failure) |
| `sd0075_boundary_replay_cases.csv` | Per-row boundary classification with first-failure category |
| `sd0075_matched_control_pairs.csv` | 7677 control rows matched to 2828 failures |
| `sd0075_matched_control_divergence_summary.json` | Per-stage failure/control rates, risk differences |
| `sd0075_counterfactual_diagnostic_cases.csv` | Before/after rows (2828=2828), improvement=737 |
| `sd0075_counterfactual_diagnostic_summary.json` | denominator_equal=true, improvement_rate=0.3050 |

---

*Report generated: 2026-05-25. All verification commands referenced in the plan were executed by the diagnostic agent.*
