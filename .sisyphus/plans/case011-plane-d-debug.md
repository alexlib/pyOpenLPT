# Case_011 Analytical Plane-d Debug and Hardening Plan

## TL;DR
> **Summary**: Debug why the analytical Stage-0 plane-offset initializer falls back with `fallback_reason=outlier_d` on `case_011`, starting with two highest-probability confirmed causes: a mismatched midpoint-vs-analytical scalar comparison and cross-window midpoint contamination. The plan also hardens the initializer for future refractive cases without changing BA parameterization.
> **Deliverables**:
> - Reproducible diagnostic evidence for `case_011` and local synthetic fixtures
> - Failing tests for parameterization mismatch and multi-window contamination
> - Targeted initializer fixes with no BA/C++ contract changes
> - Structured logging that distinguishes gate issue vs solver issue vs data issue
> - Required final confirmation via `case_011` end-to-end result with explicit `success`, `ray_mean_mm`, and `len_mean_mm`
> **Effort**: Large
> **Parallel**: YES - 3 waves
> **Critical Path**: Task 1 → Task 3 → Task 4 → Task 8 (in parallel with Task 1 → Task 5 → Task 6 → Task 7 → Task 8)

## Context
### Original Request
- Investigate why `case_011` appears to reject `d_solved` with `fallback_reason=outlier_d` for both windows.
- Determine whether the cause is a bug in the analytical solver/integration or expected real-data behavior.
- Produce a Metis-driven auto-debug plan that generalizes beyond `case_011`.

### Interview Summary
- User observed real-log evidence that the new initializer path is active because logs include both `d_midpoint` and `d_solved` plus chooser output.
- User wants a **root cause + fix path**, not a vague explanation.
- Scope should generalize beyond `case_011`, but `case_011` remains the primary reproduction case.
- Task 8 from the earlier implementation plan is not fully evidenced yet because the explicit `run_one_case(...)` success/metric checks were not captured to completion.
- User clarified that the `X_mids` fix is in scope because it belongs to plane initialization; only downstream BA redesign is out of scope.
- The case geometry / 3D cloud may provide plane-location ground truth, which should be used to compare analytical vs midpoint candidates directly.

### Metis Review (gaps addressed)
- The strongest confirmed issue is a **parameterization mismatch**: `refraction_wand_calibrator.py:486-492` computes `d0_mm` from median Euclidean midpoint range divided by `n_object`, while BA later defines plane offset as `dot(n, pt - A)` in `refraction_calibration_BA.py:417-443`.
- A second confirmed issue is **multi-window contamination**: `refraction_wand_calibrator.py:419-425` builds one global `X_mids` cloud before the per-window loop, then `:477-493` reuses it for every window when computing `n_win` and `d0_mm`.
- Existing `case_011` coverage in `tests/test_plane_d_solver.py:561-610` verifies success/metrics/log presence only; it does not capture or assert the internal diagnostics needed to distinguish gate mismatch, conditioning, or model mismatch.
- External math review supports keeping the current “thickness does not affect exit direction” reasoning for a plane-parallel plate, while still treating omitted slab-thickness translation of the ray origin as a possible real-data model gap.
- The fix boundary is the plane-initialization seam only: `refraction_wand_calibrator.py` and `plane_d_solver.py` may change, but BA/downstream optimization must remain untouched.
- If case metadata or the 3D point cloud exposes plane ground truth, compare both analytical and midpoint candidates against that truth and use it as the primary evidence for whether the analytical path is actually more accurate.

## Work Objectives
### Core Objective
Determine and fix the true cause of `outlier_d` fallback on real refractive data while preserving the Stage-0/BA contract `plane_pt = A + d * n`, the existing C++ interface, and the current automatic fallback behavior when the analytical solve is genuinely invalid.

### Deliverables
- New synthetic tests that prove or disprove the two strongest local bug hypotheses
- Synthetic truth-comparison coverage that proves the plan can tell when analytical is better, worse, or tied against midpoint
- Improved Stage-0 diagnostics for midpoint seed, analytical solution, gate threshold, and solve health
- Minimal production fix(es) in the plane-initialization seam (`refraction_wand_calibrator.py` and/or `plane_d_solver.py`), including the `X_mids` per-window fix
- Updated `case_011` verification that explicitly checks final `run_one_case(...)` success and metrics in addition to `[PLANE_INIT]`, plus truth-distance comparison between analytical and midpoint candidates when ground truth is available
- A documented decision on whether slab-thickness translation remains a known limitation or requires a follow-up implementation plan

### Definition of Done (verifiable conditions with commands)
- The root cause branch is explicitly classified as one of: `gate_mismatch`, `multi_window_seed_bug`, `data_subset_issue`, `model_gap`, or `expected_real_data_fallback`.
- There is at least one failing synthetic/local test that reproduces the confirmed bug before the fix and passes after the fix.
- There is at least one synthetic truth-distance test that proves the plan can detect when analytical is better than midpoint and when midpoint is better than analytical.
- `PlaneInitializer.init_window_planes_from_cameras(...)` logs enough information to explain any future `outlier_d` rejection without additional code spelunking.
- `conda run -n OpenLPT python -m pytest tests/test_plane_d_solver.py tests/test_plane_pipeline.py -v` passes.
- Required final confirmation: `case_011` `run_one_case(...)` completes with explicit evidence captured.
- If ground truth plane location is available, the plan records whether analytical or midpoint is closer to truth and uses that to decide whether the analytical seed is actually better.
- If `case_011` still falls back after the fixes, the log and tests prove **why** that fallback is correct.

### Must Have
- Preserve BA parameterization and downstream plane bookkeeping in `refraction_calibration_BA.py:417-443`.
- Preserve Stage-0 public output contract per window: `{'plane_pt', 'plane_n', 'thick_mm', 'initialized'}`.
- Preserve default-on analytical attempt with automatic fallback.
- Use the `OpenLPT` conda environment for all verification commands.
- Add structured evidence for: `d_midpoint`, `d_projected` (if added), `d_solved`, `delta`, threshold, `n_equations`, `rank`, `cond`, `residual_rms`, and `camera_side_ok`.
- Keep behavior changes inside the plane-initialization seam only; calibration code may change when it is part of that seam, but downstream BA must stay untouched.
- When truth is available, compare both candidates against it and record the delta; this is the strongest discriminator for whether analytical actually improves initialization.

### Must NOT Have
- Do **not** change `RefractiveBAOptimizer`, `_plane_anchor`, `_plane_d0`, or C++ camera/export logic.
- Do **not** change `plane_pt = A + d * n` parameterization.
- Do **not** hard-code case-specific constants from `case_011` into production logic or tests.
- Do **not** combine threshold-tuning with formula/bug fixes in the same task; isolate causality.
- Do **not** expand this work into a full thick-plate physical-model rewrite unless the diagnostics prove it is the blocking root cause.

## Verification Strategy
> ZERO HUMAN INTERVENTION - all verification is agent-executed.
- Test decision: tests-after with explicit failing-test-first steps on local synthetic fixtures, then full regression, then external `case_011`
- QA policy: Every task includes agent-executed scenarios
- Evidence: `.sisyphus/evidence/task-{N}-{slug}.{ext}`
- Priority of truth signals:
  1. synthetic/local failing tests that isolate bug hypotheses
  2. ground-truth comparison of analytical vs midpoint candidates
  3. Stage-0 structured diagnostics for real data
  4. subset-stability checks (camera-pair and frame-block)
  5. only if needed for final validation, `case_011` `run_one_case(...)` result metrics
- `case_011` final validation remains optional and only if diagnosis needs a live end-to-end confirmation:
  - `success=True`
  - `ray_mean_mm <= 0.002`
  - `len_mean_mm <= 0.005`
  - `[PLANE_INIT]` present with per-window chooser output

### Metis Auto-Debug Loop
> If the analytical solver is worse than midpoint on truth-distance, do not guess. Loop.
1. Run the current synthetic verification set and ground-truth comparison on available case data.
2. Compare analytical and midpoint candidates against truth when truth is available.
3. If analytical is worse than midpoint, or if `outlier_d` still rejects an analytically better candidate, collect the exact diagnostics: `d_midpoint`, `d_solved`, truth-distance, `n_equations`, `rank`, `cond`, `residual_rms`, `camera_side_ok`, and the per-window midpoint/cloud scope.
4. Ask Metis to classify the failure as one of: gate mismatch, shared-seed bug, subset instability, or true model gap.
5. Convert Metis’ answer into a single worker-fix task inside the plane-initialization seam.
6. Rerun the same verification set.
7. Repeat until either analytical is demonstrably closer to truth than midpoint, or the fallback is proven correct and the evidence says so.
8. Never close the plan while truth-distance evidence still contradicts the chosen initializer.

## Execution Strategy
### Parallel Execution Waves
> Target: 5-8 tasks per wave. <3 per wave (except final) = under-splitting.
> Extract shared dependencies as Wave-1 tasks for max parallelism.

Wave 1: reference audit + external `case_011` artifact capture

Wave 2: failing tests for scalar mismatch and shared `X_mids` contamination

Wave 3: initializer fixes and gate diagnostics

Wave 4: discriminator tests for threshold / stability / truth-distance

Wave 5: final `case_011` verification and outcome classification

### Dependency Matrix (full, all tasks)
- Task 1 blocks Tasks 3-6
- Task 2 blocks Tasks 7-8
- Task 3 blocks Tasks 4 and 8
- Task 4 blocks Task 8
- Task 5 blocks Tasks 6 and Task 8
- Task 6 blocks Tasks 7-8
- Task 7 blocks Task 8
- Task 8 depends on Tasks 2-7

### Agent Dispatch Summary (wave → task count → categories)
- Wave 1 → 2 tasks → `quick`, `deep`
- Wave 2 → 2 tasks → `quick`, `quick`
- Wave 3 → 2 tasks → `unspecified-low`, `unspecified-low`
- Wave 4 → 1 task → `deep`
- Wave 5 → 1 task → `deep`

## TODOs
> Implementation + Test = ONE task. Never separate.
> EVERY task MUST have: Agent Profile + Parallelization + QA Scenarios.

- [ ] 1. Audit and lock the exact Stage-0 scalar definitions before changing behavior

  **What to do**: Trace every local use of `d0_mm`, `d_midpoint`, `X_mids`, and the BA plane offset definition so the executor can prove the parameterization mismatch before editing code. Specifically verify `refraction_wand_calibrator.py:419-425`, `:477-493`, `:554-587`, and `refraction_calibration_BA.py:417-443`, then capture a short engineering note in code comments/tests or debug notepad describing the current scalar meanings: Euclidean midpoint range divided by `n_object` vs anchor-normal plane offset.
  **Must NOT do**: Do not change runtime behavior in this task. Do not tune thresholds. Do not touch C++/BA files beyond read-only confirmation.

  **Recommended Agent Profile**:
  - Category: `quick` - Reason: local reference audit and invariant capture
  - Skills: `[]` - repo-local tracing only
  - Omitted: `['brainstorming']` - the hypotheses are already ranked

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: [3, 4, 5, 6] | Blocked By: []

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `modules/camera_calibration/wand_calibration/refraction_wand_calibrator.py:419-425` - global `X_mids` construction before per-window loop
  - Pattern: `modules/camera_calibration/wand_calibration/refraction_wand_calibrator.py:477-493` - current `n_win`, Euclidean `depth_med`, `n_object`, and `d0_mm` computation
  - Pattern: `modules/camera_calibration/wand_calibration/refraction_wand_calibrator.py:554-587` - analytical solve call, chooser logic, and `[PLANE_INIT]` log
  - API/Type: `modules/camera_calibration/wand_calibration/refraction_calibration_BA.py:417-443` - downstream BA offset definition `dot(n0, pt0 - A)`
  - Test: `tests/test_plane_d_solver.py:344-375` - current `outlier_d` test coverage is too coarse

  **Acceptance Criteria** (agent-executable only):
  - [ ] All local definitions of Stage-0 `d` and BA `d` are documented in one place for the executor.
  - [ ] The executor can answer, with file references, whether the current gate compares like-for-like scalars.
  - [ ] `conda run -n OpenLPT python -m pytest tests/test_plane_d_solver.py -v -k outlier` still passes unchanged after the audit-only task.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: Reference audit does not change behavior
    Tool: Bash
    Steps: Run `conda run -n OpenLPT python -m pytest tests/test_plane_d_solver.py -v -k outlier`
    Expected: PASS; audit-only task introduces no functional change
    Evidence: .sisyphus/evidence/task-1-audit.txt

  Scenario: Executor can trace both scalar definitions exactly
    Tool: Bash
    Steps: Run a targeted grep/read verification on the four referenced line ranges and capture the output in evidence
    Expected: PASS; evidence shows Euclidean midpoint formula in Stage-0 and projection formula in BA
    Evidence: .sisyphus/evidence/task-1-audit-refs.txt
  ```

  **Commit**: NO | Message: `n/a` | Files: [audit only]

- [ ] 2. Capture reproducible `case_011` diagnostics and ground-truth comparison artifacts

  **What to do**: Prefer the existing `case_011` log and any available case metadata / 3D cloud artifacts to compare analytical vs midpoint against ground truth. If full `run_one_case(...)` validation is already cheap enough in the environment, it may be used as a final confirmation step; otherwise do not block diagnosis on a full calibration run. Capture explicit evidence for the available truth-distance comparison, `[PLANE_INIT]` lines, and any persisted log/result artifact paths that are already present.
  If case metadata or reconstructed 3D points expose plane ground truth, compute and record the truth-distance of both the analytical candidate and midpoint fallback for each window; this is the key evidence for whether analytical is genuinely more accurate.
  If no ground truth artifact is available, explicitly record that fact and continue with synthetic truth-comparison evidence from Task 3/7.
  **Must NOT do**: Do not rely on partial console output. Do not require a full `run_one_case(...)` calibration run for diagnosis. Do not change production logic in this task.

  **Recommended Agent Profile**:
  - Category: `deep` - Reason: external harness evidence gathering and long-running verification
  - Skills: `[]` - command execution and artifact inspection only
  - Omitted: `['proactive-verify']` - verification steps are fully specified here

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: [7, 8] | Blocked By: []

  **References** (executor has NO interview context - be exhaustive):
  - External: `J:/Refraction_test/test_script/run_calibration_worker.py` - source of `run_one_case(...)`
  - External: `J:/Refraction_test/case_011` - primary repro dataset
  - External: `J:/Refraction_test/test_results/logs/robustness/case_011.log` - persisted Stage-0/BA log path already observed in this session
  - Test: `tests/test_plane_d_solver.py:561-610` - existing `case_011` test currently checks only success/metrics/log-presence
  - Command: `conda run -n OpenLPT python -c "import sys; sys.path.insert(0, r'J:/Refraction_test/test_script'); from run_calibration_worker import run_one_case; result = run_one_case(r'J:/Refraction_test/case_011', r'J:/Refraction_test/test_results'); print(result['success']); print(result['metrics']); print(result['log_path'])"`

  **Acceptance Criteria** (agent-executable only):
  - [ ] Evidence includes ground-truth comparison of analytical vs midpoint candidates when such truth artifacts are available.
  - [ ] Evidence includes every `[PLANE_INIT]` line needed to explain the initializer choice.
  - [ ] If a full `run_one_case(...)` validation is performed, it is explicitly documented as confirmation only and not required for diagnosis.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: Ground-truth comparison is captured without full calibration
    Tool: Bash
    Steps: Read the existing case_011 log and available case metadata / 3D cloud artifact to compare analytical vs midpoint truth-distance
    Expected: PASS; evidence contains the per-window truth-distance comparison or an explicit GT-unavailable note
    Evidence: .sisyphus/evidence/task-2-case011-truth.txt

  Scenario: PLANE_INIT evidence is complete per window
    Tool: Bash
    Steps: Search the final case_011 log for every `[PLANE_INIT]`, `[WIN_SANITY]`, and any solver-verbose lines around them
    Expected: PASS; evidence shows one complete initialization block per window with chooser output
    Evidence: .sisyphus/evidence/task-2-case011-log.txt
  ```

  **Commit**: NO | Message: `n/a` | Files: [verification only]

- [ ] 3. Add a failing synthetic test for scalar-definition mismatch in the `outlier_d` gate

  **What to do**: Create a local synthetic fixture where the correct plane offset along `plane_n` differs significantly from Euclidean camera-to-midpoint range, while the analytical solve remains low-residual and physically valid. The test must prove that comparing `d_solved` to `median(||X_mid - C_mean||)/n_object` is not a like-for-like comparison. Use this fixture to fail under the current code before any fix. The preferred assertion is that a projection-consistent midpoint seed would remain within gate tolerance while the current Euclidean seed incorrectly triggers `outlier_d`.
  **Must NOT do**: Do not use `case_011` or any J-drive dependency. Do not weaken the test to only check log presence. Do not modify production code in this task.

  **Recommended Agent Profile**:
  - Category: `quick` - Reason: isolated failing test around one mathematical inconsistency
  - Skills: `[]` - synthetic fixture authoring only
  - Omitted: `['verification-before-completion']` - broader regression is covered later

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: [4, 8] | Blocked By: [1]

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `tests/test_plane_d_solver.py:344-375` - current outlier gate test uses only an extreme fake seed (`5000.0`), not a realistic mismatched scalar
  - Pattern: `tests/test_plane_d_solver.py:413-476` - required solver return keys and deterministic structure
  - Pattern: `modules/camera_calibration/wand_calibration/refraction_wand_calibrator.py:486-492` - current Euclidean midpoint formula to challenge
  - API/Type: `modules/camera_calibration/wand_calibration/refraction_calibration_BA.py:417-443` - target scalar definition to align with

  **Acceptance Criteria** (agent-executable only):
  - [ ] A new test fails on the current code and demonstrates the gate compares unlike quantities.
  - [ ] The failure message clearly shows current seed, projection-consistent seed, and `d_solved`.
  - [ ] `conda run -n OpenLPT python -m pytest tests/test_plane_d_solver.py -v -k scalar_definition` fails before the fix and passes after the fix.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: Mismatched scalar definition reproduces false outlier rejection
    Tool: Bash
    Steps: Run `conda run -n OpenLPT python -m pytest tests/test_plane_d_solver.py -v -k scalar_definition`
    Expected: FAIL on current code with an assertion showing Euclidean midpoint seed is not comparable to analytical `d`
    Evidence: .sisyphus/evidence/task-3-scalar-definition-fail.txt

  Scenario: Test remains local and deterministic
    Tool: Bash
    Steps: Re-run the same test after fixture creation without external data
    Expected: Same deterministic failure signature before fix
    Evidence: .sisyphus/evidence/task-3-scalar-definition-repeat.txt
  ```

  **Commit**: YES | Message: `test(wand-cal): expose scalar-definition mismatch in plane-d gate` | Files: [`tests/test_plane_d_solver.py`]

- [ ] 4. Replace the midpoint seed comparison with a projection-consistent scalar for gate evaluation

  **What to do**: Update `PlaneInitializer.init_window_planes_from_cameras(...)` so the midpoint-based reference scalar used for `outlier_d` comparison is computed in the **same parameterization** as BA and the analytical solver: along `plane_n` from `C_mean`/`A_anchor`, not by Euclidean midpoint range divided by `n_object`. The fix must preserve current orientation logic, closest-interface semantics, and `plane_pt = C_mean + d * n_win`. Remove the `/ n_object` scaling and pass a like-for-like reference scalar into `solve_plane_d_from_correspondences(...)`.
  **Must NOT do**: Do not tune the 50% threshold in this task. Do not change BA. Do not change the solver’s own `d` parameterization. Do not remove fallback behavior.

  **Recommended Agent Profile**:
  - Category: `unspecified-low` - Reason: targeted integration fix with contract preservation
  - Skills: `[]` - local refactor only
  - Omitted: `['systematic-debugging']` - the debug branch is already defined

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: [8] | Blocked By: [1, 3]

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `modules/camera_calibration/wand_calibration/refraction_wand_calibrator.py:470-493` - current `C_mean`, `n_win`, `depth_med`, `n_object`, `d0_mm`
  - Pattern: `modules/camera_calibration/wand_calibration/refraction_wand_calibrator.py:561-573` - current solver call receives `d_midpoint=d0_mm`
  - API/Type: `modules/camera_calibration/wand_calibration/refraction_calibration_BA.py:417-443` - required scalar definition `dot(n0, pt0 - A)`
  - Test: `tests/test_plane_d_solver.py` - Task 3 failing test must be the acceptance driver

  **Acceptance Criteria** (agent-executable only):
  - [ ] The scalar used for midpoint-vs-analytical comparison is projection-consistent with BA/solver `d`.
  - [ ] Task 3’s failing test now passes.
  - [ ] Existing `outlier_d` rejection behavior for obviously absurd seeds (e.g. `5000.0`) still passes.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: Projection-consistent midpoint seed fixes false outlier
    Tool: Bash
    Steps: Run `conda run -n OpenLPT python -m pytest tests/test_plane_d_solver.py -v -k "scalar_definition or outlier_d_rejected"`
    Expected: PASS; scalar-definition test passes and absurd-seed test still rejects correctly
    Evidence: .sisyphus/evidence/task-4-scalar-fix.txt

  Scenario: No BA contract regression
    Tool: Bash
    Steps: Run targeted integration tests that call `PlaneInitializer.init_window_planes_from_cameras(...)`
    Expected: PASS; returned `plane_pt`/`plane_n` structure unchanged and `[PLANE_INIT]` still emitted
    Evidence: .sisyphus/evidence/task-4-scalar-fix-integration.txt
  ```

  **Commit**: YES | Message: `fix(wand-cal): align midpoint gate with plane offset parameterization` | Files: [`modules/camera_calibration/wand_calibration/refraction_wand_calibrator.py`, `tests/test_plane_d_solver.py`]

- [ ] 5. Add a failing multi-window test that exposes shared `X_mids` contamination across windows

  **What to do**: Add a synthetic 2-window fixture where `cam_to_window={0:0,1:0,2:1,3:1}` and the midpoint clouds for the two windows are intentionally different. The test must prove that computing `X_mids` once and reusing it across windows can distort either `n_win`, the midpoint seed, orientation scoring, or chooser output. Prefer an assertion on distinct per-window seeds/logs and on the chosen plane geometry for each window.
  **Must NOT do**: Do not use external files. Do not write a test that passes even if `X_mids` remains global. Do not modify runtime code in this task.

  **Recommended Agent Profile**:
  - Category: `quick` - Reason: isolated synthetic regression for a multi-window bug candidate
  - Skills: `[]` - local pytest only
  - Omitted: `['requesting-code-review']` - not needed at task granularity

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: [6, 8] | Blocked By: [1]

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `modules/camera_calibration/wand_calibration/refraction_wand_calibrator.py:419-425` - global `X_mids` build site
  - Pattern: `modules/camera_calibration/wand_calibration/refraction_wand_calibrator.py:439-552` - per-window loop that currently reuses global midpoint cloud
  - Test: `tests/test_plane_d_solver.py:479-556` - integration test style for `PlaneInitializer`
  - Test: `tests/test_plane_pipeline.py:624-679` - live initializer output assertions and monkeypatch patterns

  **Acceptance Criteria** (agent-executable only):
  - [ ] A new multi-window synthetic test fails on the current code and localizes the issue to shared `X_mids` reuse.
  - [ ] The failure message identifies the incorrect shared-seed behavior.
  - [ ] `conda run -n OpenLPT python -m pytest tests/test_plane_d_solver.py tests/test_plane_pipeline.py -v -k multi_window_seed` fails before the fix and passes after the fix.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: Multi-window shared midpoint cloud causes incorrect per-window seeds
    Tool: Bash
    Steps: Run `conda run -n OpenLPT python -m pytest tests/test_plane_d_solver.py tests/test_plane_pipeline.py -v -k multi_window_seed`
    Expected: FAIL on current code with evidence that both windows reuse the same midpoint cloud
    Evidence: .sisyphus/evidence/task-5-multiwindow-fail.txt

  Scenario: Failure is independent of case_011 data availability
    Tool: Bash
    Steps: Re-run the same local synthetic test with no J-drive dependency
    Expected: Same deterministic failure before fix
    Evidence: .sisyphus/evidence/task-5-multiwindow-repeat.txt
  ```

  **Commit**: YES | Message: `test(wand-cal): expose shared midpoint contamination across windows` | Files: [`tests/test_plane_d_solver.py`, `tests/test_plane_pipeline.py`]

- [ ] 6. Compute midpoint references per window and add gate-decomposition diagnostics

  **What to do**: Refactor `PlaneInitializer.init_window_planes_from_cameras(...)` so each window computes or filters its own midpoint cloud before deriving `X_centroid`, orientation stats, and the midpoint reference scalar. Prefer filtering by camera-observation visibility for the current window; if that is not available, build the midpoint cloud inside the per-window loop so it is at least window-scoped during initialization. In the same task, add deterministic diagnostics that decompose the gate decision: `d_midpoint_likeforlike`, optional raw `d_euclidean`, `d_solved`, `abs_delta`, `gate_threshold`, `delta_ratio`, `n_equations`, `rank`, `cond`, `residual_rms`, and `camera_side_ok`. These diagnostics must appear in Stage-0 logs and/or a structured return path used by tests, without changing the public window-plane output contract.
  **Must NOT do**: Do not change the output dict schema for `window_planes`. Do not change solver math here unless the per-window seed fix proves insufficient. Do not remove current `[PLANE_INIT]` logging; extend it.

  **Recommended Agent Profile**:
  - Category: `unspecified-low` - Reason: localized integration refactor plus observability enhancement
  - Skills: `[]` - local integration only
  - Omitted: `['webapp-testing']` - not a UI flow

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: [7, 8] | Blocked By: [1, 5]

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `modules/camera_calibration/wand_calibration/refraction_wand_calibrator.py:419-425` - global midpoint construction to replace
  - Pattern: `modules/camera_calibration/wand_calibration/refraction_wand_calibrator.py:506-524` - orientation scoring currently uses `X_mids[:200]`
  - Pattern: `modules/camera_calibration/wand_calibration/refraction_wand_calibrator.py:603-607` - `[WIN_SANITY]` object-side percentage also uses `X_mids[:200]`
  - Pattern: `modules/camera_calibration/wand_calibration/refraction_wand_calibrator.py:584-607` - current `[PLANE_INIT]` and `[WIN_SANITY]` logging
  - API/Type: `modules/camera_calibration/wand_calibration/plane_d_solver.py:243-275` - already returns `rank`, `cond`, `n_equations`, `residual_rms`, `camera_side_ok`
  - Test: `tests/test_plane_d_solver.py:523-556` - existing integration log assertions are the starting point

  **Acceptance Criteria** (agent-executable only):
  - [ ] Task 5’s failing multi-window test now passes.
  - [ ] Logs or test-accessible diagnostics expose the exact reason for any future `outlier_d` rejection.
  - [ ] `conda run -n OpenLPT python -m pytest tests/test_plane_d_solver.py tests/test_plane_pipeline.py -v -k "multi_window_seed or PLANE_INIT or integration"` passes.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: Per-window midpoint computation eliminates shared-cloud contamination
    Tool: Bash
    Steps: Run `conda run -n OpenLPT python -m pytest tests/test_plane_d_solver.py tests/test_plane_pipeline.py -v -k "multi_window_seed or integration"`
    Expected: PASS; per-window synthetic tests pass and integration log checks still succeed
    Evidence: .sisyphus/evidence/task-6-multiwindow-fix.txt

  Scenario: Gate decomposition diagnostics are emitted deterministically
    Tool: Bash
    Steps: Run the same subset with log capture enabled
    Expected: PASS; logs include delta, threshold, and solver health fields whenever analytical solve is attempted
    Evidence: .sisyphus/evidence/task-6-diagnostics.txt
  ```

  **Commit**: YES | Message: `fix(wand-cal): compute per-window midpoint seeds and log gate diagnostics` | Files: [`modules/camera_calibration/wand_calibration/refraction_wand_calibrator.py`, `modules/camera_calibration/wand_calibration/plane_d_solver.py`, `tests/test_plane_d_solver.py`, `tests/test_plane_pipeline.py`]

- [ ] 7. Add discriminator tests for threshold boundary, subset stability, and model-gap triage

  **What to do**: Add focused tests or scripted checks that distinguish four remaining branches after Tasks 4 and 6: (a) the gate threshold itself is too strict, (b) one camera pair/frame subset drives instability, (c) the analytical solution is genuinely worse than midpoint on a given geometry, or (d) the remaining discrepancy is a physical/model gap such as omitted slab-thickness translation. Include boundary tests around the current 50% threshold, at least one subset-stability harness that solves using per-camera-pair or leave-one-camera-out subsets on local synthetic data, and at least one truth-distance fixture that proves whether analytical or midpoint is closer to the known plane truth. If slab-thickness translation is still unproven, document it as a follow-up branch rather than changing solver math in this task.
  **Must NOT do**: Do not silently tune the threshold without a test that proves why. Do not start a thick-plate rewrite unless the new evidence clearly points there. Do not rely only on condition number for stability; use subset-to-subset drift.

  **Recommended Agent Profile**:
  - Category: `deep` - Reason: branch-discriminator work with multiple competing hypotheses
  - Skills: `[]` - local experimentation and tests only
  - Omitted: `['artistry']` - conventional debugging path is sufficient

  **Parallelization**: Can Parallel: YES | Wave 4 | Blocks: [8] | Blocked By: [4, 6]

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `modules/camera_calibration/wand_calibration/plane_d_solver.py:208-239` - current fallback gating logic
  - Pattern: `modules/camera_calibration/wand_calibration/plane_d_solver.py:149-203` - pairwise equation assembly and least-squares solve
  - Test: `tests/test_plane_d_solver.py:380-409` - current ill-conditioned coverage is minimal
  - Research note: subset-to-subset drift is a stronger stability discriminator than `cond(Nx1)` alone for this solver shape
  - Research note: truth-distance comparison is the strongest discriminator for whether analytical is better or worse than midpoint

  **Acceptance Criteria** (agent-executable only):
  - [ ] Threshold boundary behavior is covered by deterministic tests.
  - [ ] There is an automated way to tell stable-but-rejected from unstable analytical solves.
  - [ ] There is an automated truth-distance comparison that can prove analytical is better or worse than midpoint on a known plane fixture.
  - [ ] Any remaining slab-thickness concern is either disproven locally or explicitly recorded as a follow-up branch with evidence.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: Gate boundary tests distinguish just-below and just-above threshold
    Tool: Bash
    Steps: Run `conda run -n OpenLPT python -m pytest tests/test_plane_d_solver.py -v -k gate_boundary`
    Expected: PASS; one fixture is accepted just below threshold and another rejected just above it
    Evidence: .sisyphus/evidence/task-7-gate-boundary.txt

  Scenario: Subset stability harness classifies stable vs unstable solves
    Tool: Bash
    Steps: Run `conda run -n OpenLPT python -m pytest tests/test_plane_d_solver.py -v -k subset_stability`
    Expected: PASS; output distinguishes low-drift stable solves from subset-sensitive unstable solves
    Evidence: .sisyphus/evidence/task-7-subset-stability.txt
  ```

  **Commit**: YES | Message: `test(wand-cal): add gate boundary and subset stability coverage` | Files: [`tests/test_plane_d_solver.py`]

- [ ] 8. Re-run full regression and classify the final `case_011` outcome explicitly

  **What to do**: After Tasks 4, 6, and 7, run the full local regression suite. Then classify the final outcome in one of two ways: (1) analytical init is now accepted on at least one window for a justified reason and all final metrics remain within thresholds, or (2) fallback still occurs but the new diagnostics prove it is correct (e.g. unstable subsets, legitimate gate exceedance, or residual/model gap). If the current `case_011` assertions are insufficient, record the exact assertion gap and fold the strengthening into a follow-up test task before closing the plan. If optional confirmation is warranted, the real `case_011` harness may be used, but it is not required for diagnosis.
  If truth is available, report whether analytical or midpoint is closer to the ground-truth plane location; use that as the primary quality signal for the analytical solver.
  **Must NOT do**: Do not claim success from partial logs. Do not mark the plan complete unless the required evidence for the chosen validation path is explicitly captured; final confirmation requires `success`, `ray_mean_mm`, `len_mean_mm`, and the per-window Stage-0 classification.

  **Recommended Agent Profile**:
  - Category: `deep` - Reason: final evidence synthesis across local and external verification
  - Skills: `[]` - verification and classification only
  - Omitted: `['prepare-pr']` - packaging work is out of scope

  **Parallelization**: Can Parallel: NO | Wave 5 | Blocks: [Final Verification Wave] | Blocked By: [2, 3, 4, 5, 6, 7]

  **References** (executor has NO interview context - be exhaustive):
  - Command: `conda run -n OpenLPT python -m pytest tests/test_plane_d_solver.py tests/test_plane_pipeline.py -v`
  - Command: `conda run -n OpenLPT python -c "import sys; sys.path.insert(0, r'J:/Refraction_test/test_script'); from run_calibration_worker import run_one_case; result = run_one_case(r'J:/Refraction_test/case_011', r'J:/Refraction_test/test_results'); print(result['success']); print(result['metrics']); print(result['log_path'])"`
  - External: `J:/Refraction_test/test_results/logs/robustness/case_011.log` - final Stage-0 and BA evidence source
  - Test: `tests/test_plane_d_solver.py:561-610` - current case test to strengthen after the debug branch is settled

  **Acceptance Criteria** (agent-executable only):
  - [ ] `tests/test_plane_d_solver.py tests/test_plane_pipeline.py` pass.
  - [ ] If optional confirmation is used, `run_one_case(...)` returns `success=True` and `ray_mean_mm <= 0.002`, `len_mean_mm <= 0.005`.
  - [ ] Final evidence explicitly states whether `case_011` analytical init is accepted or correctly rejected, window by window.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: Full regression passes after debug fixes
    Tool: Bash
    Steps: Run `conda run -n OpenLPT python -m pytest tests/test_plane_d_solver.py tests/test_plane_pipeline.py -v`
    Expected: PASS; local synthetic and integration tests all succeed
    Evidence: .sisyphus/evidence/task-8-full-regression.txt

  Scenario: case_011 final classification is explicit when optional confirmation is used
    Tool: Bash
    Steps: If optional confirmation is needed, run the documented `run_one_case(...)` command, then parse the resulting log for `[PLANE_INIT]`, `[WIN_SANITY]`, and final metric lines
    Expected: PASS; evidence includes `success=True`, compliant metrics, and a per-window explanation of accept vs fallback
    Evidence: .sisyphus/evidence/task-8-case011-final.txt
  ```

  **Commit**: NO | Message: `n/a` | Files: [verification only]

## Final Verification Wave (MANDATORY — after ALL implementation tasks)
> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.
> **Do NOT auto-proceed after verification. Wait for user's explicit approval before marking work complete.**
> **Never mark F1-F4 as checked before getting user's okay.** Rejection or user feedback -> fix -> re-run -> present again -> wait for okay.
- [ ] F1. Plan Compliance Audit — oracle
- [ ] F2. Code Quality Review — unspecified-high
- [ ] F3. Real Manual QA — unspecified-high (+ playwright if UI)
- [ ] F4. Scope Fidelity Check — deep

## Commit Strategy
- Commit 1: `test(wand-cal): expose scalar-definition mismatch in plane-d gate`
- Commit 2: `fix(wand-cal): align midpoint gate with plane offset parameterization`
- Commit 3: `test(wand-cal): expose shared midpoint contamination across windows`
- Commit 4: `fix(wand-cal): compute per-window midpoint seeds and log gate diagnostics`
- Commit 5: `test(wand-cal): add gate boundary and subset stability coverage`

## Success Criteria
- The executor can prove whether the original `outlier_d` fallback was caused by a real bug, a gate mismatch, or legitimate geometry/model limitations.
- The midpoint comparison scalar and analytical solver scalar are in the same parameterization.
- Multi-window cases no longer reuse one global midpoint cloud when initializing per-window planes.
- Future `outlier_d` rejections are diagnosable from logs/tests without repeating this investigation.
- `case_011` has explicit verified evidence for `success`, `ray_mean_mm`, `len_mean_mm`, and per-window Stage-0 classification in final confirmation.
