# Correspondence-Informed Refractive Plane Initializer

## TL;DR
> **Summary**: Add a default-on, analytical Stage-0 refractive plane initializer that solves the **plane location / scalar offset `d` (hence `plane_pt`) from 2D correspondences** while preserving the current normal estimate, known thickness, and automatic fallback to the legacy midpoint-depth initializer.
> **Deliverables**:
> - New analytical solver for fixed-normal plane offset `d` from 2D correspondences
> - Stage-0 integration in the current plane-init seam before C++ camera creation and BA
> - Unit, seam, and `case_011` verification coverage
> - Diagnostic logging that reports analytical-`d` vs midpoint-depth candidate selection
> **Effort**: Medium
> **Parallel**: YES - 2 waves
> **Critical Path**: Task 1 → Task 3 → Task 4 → Task 5 → Task 8

## Context
### Original Request
- Evaluate whether the literature-inspired correspondence-based refractive plane initialization method is feasible in the current OpenLPT refractive wand calibration workflow.
- If feasible, write an implementation plan.
- Use `J:\Refraction_test\case_011` for verification and inspect `J:\Refraction_test\test_script\run_robustness.py` / `run_calibration_worker.py` to understand execution.

### Interview Summary
- User wants the new initializer to run **by default with automatic fallback** to the current midpoint initializer.
- Success is defined as **better pre-BA initialization quality** on `case_011`; final BA metrics may improve modestly or stay similar as long as initialization quality is measurably better and there is no regression.
- Plane thickness is known and should stay fixed.
- The user asked for feasibility review before planning; direct code implementation is explicitly out of scope for this session.

### Metis Review (gaps addressed)
- A direct closed-form port of Chen & Yang (CVPR 2014) / Agrawal et al. (CVPR 2012) is **not** a clean fit for this wand-endpoint workflow.
- The feasible reduced scope is: solve a better **plane location / `plane_pt` (via scalar offset `d`)** from 2D correspondences while keeping the current normal estimate, preserving closest-interface semantics, and adding **automatic fallback** when the solve is ill-conditioned or produces a worse seed.
- The clean integration seam is the existing Stage-0 call to `PlaneInitializer.init_window_planes_from_cameras(...)` in `modules/camera_calibration/wand_calibration/refraction_wand_calibrator.py:2208-2213`.
- Verification should use `J:\Refraction_test\test_script\run_calibration_worker.py::run_one_case(...)`, parse the generated log, compare solved `d` against synthetic ground truth where available, and treat Stage-0 diagnostics plus early BA improvements as the primary acceptance signal.

## Work Objectives
### Core Objective
Add a Stage-0 plane initializer that analytically solves the initial window plane location from 2D multi-view correspondences with fixed `plane_n`, then hands a fully compatible `window_planes` structure into the existing refractive BA pipeline.

### Deliverables
- `modules/camera_calibration/wand_calibration/plane_d_solver.py` (or equivalent geometry-local solver module)
- Updated Stage-0 integration in `modules/camera_calibration/wand_calibration/refraction_wand_calibrator.py`
- New unit/integration tests in `tests/test_plane_d_solver.py`
- Targeted seam regression updates in `tests/test_plane_pipeline.py`
- A `case_011` end-to-end verification test that compares default-on behavior against a forced-legacy baseline

### Definition of Done (verifiable conditions with commands)
- The default calibration flow uses the new initializer path before BA and logs a winner line for each initialized window.
- When geometry is weak or the new candidate is not clearly better, the code falls back to the current midpoint initializer without changing downstream contracts.
- Existing plane-init seam tests still pass.
- New helper/unit tests pass.
- `case_011` runs successfully through `run_one_case(...)` and remains non-regressed on final ray/length metrics while showing valid Stage-0 diagnostics.
- Commands:
  - `conda run -n OpenLPT python -m pytest tests/test_plane_d_solver.py -v`
  - `conda run -n OpenLPT python -m pytest tests/test_plane_pipeline.py -v -k correspondence`
  - `conda run -n OpenLPT python -c "import sys; sys.path.insert(0, r'J:/Refraction_test/test_script'); from run_calibration_worker import run_one_case; result = run_one_case(r'J:/Refraction_test/case_011', r'J:/Refraction_test/test_results'); print(result['success']); print(result['metrics'])"`

### Must Have
- Preserve the public Stage-0 output contract: `{'plane_pt', 'plane_n', 'thick_mm', 'initialized'}` per window.
- Preserve current `plane_n` orientation and closest-interface semantics, but replace the legacy `d0_mm = median_depth / n_object` seed with an analytical 2D-correspondence solve for `d` when it is valid.
- Preserve closest-interface semantics and camera-side sign convention.
- Keep known thickness fixed from `window_media`.
- Make the new initializer default-on with silent fallback.
- Emit deterministic diagnostics showing midpoint-depth candidate, analytical `d` candidate, winner, and fallback reason when applicable.

## Solver Appendix (implementation contract)
### Parameterization
- Use the same per-window anchor as BA: `A = mean(camera_centers_for_window)`.
- Represent the plane by
  
  `plane_pt = A + d * plane_n`
  
  where `plane_n` is the current Stage-0 normal seed and `d` is the only solved scalar.

### Per-correspondence equation
For each 3D wand endpoint observed in at least two cameras `j,k`:
1. Build camera rays
   
   `u_cam = normalize(K_j^{-1} [u, v, 1]^T)`
   
   `R_j = Rodrigues(cam_params[cid][0:3])`
   
   `u_ij = normalize(R_j^T * u_cam)`
2. Write the first interface intersection for camera `j` as
   
   `Q_ij(d) = C_j + (((plane_n^T A) + d - plane_n^T C_j) / (plane_n^T u_ij)) * u_ij`
   
   and equivalently
   
   `Q_ij(d) = Q_ij0 + d * q_ij`, where `Q_ij0 = C_j + ((plane_n^T A - plane_n^T C_j)/(plane_n^T u_ij)) * u_ij` and `q_ij = u_ij / (plane_n^T u_ij)`.
3. Compute the refracted ray direction `v_ij` using the existing two-interface Snell model with fixed `plane_n` and known thickness `h`.
4. For the pair `(j,k)`, form
   
   `c_i = v_ij × v_ik`
   
   and the scalar constraint
   
   `r_i(d) = (Q_ij(d) - Q_ik(d)) · c_i = 0`
5. Expand to a linear equation in `d`:
   
   `A_i = (q_ij - q_ik) · c_i`
   
   `b_i = -((Q_ij0 - Q_ik0) · c_i)`
   
   so each pair contributes `A_i d = b_i`.

### Stacking rule
- For a given endpoint, form pairwise equations from **all valid camera pairs** that observe that endpoint and whose rays are not near-parallel to the plane.
- Stack all equations from all valid endpoint pairs into one least-squares system `A d = b`.
- Use `np.linalg.lstsq` to solve for `d_solved`.

### Validity / fallback thresholds
- Minimum accepted system size: at least 2 independent pairwise equations total.
- Reject if `cond(A) > 1e8` or if `A` is rank-deficient.
- Reject if `d_solved` is non-finite, or if `|d_solved - d_midpoint| > 0.5 * max(|d_midpoint|, 1.0)`.
- Reject if the resulting `plane_pt` violates camera-side checks for any active camera.
- On rejection, fall back to the legacy midpoint-depth seed exactly as currently computed.

### Solver API contract
- Suggested function signature:
  
  `solve_plane_d_from_correspondences(cam_params, observations, plane_n, window_media, cam_to_window, active_cam_ids=None, verbose=False) -> dict`
- `cam_params`: per-camera bootstrap extrinsics/intrinsics in the same format already used by Stage-0.
- `cam_params[cid]` is the existing 11-element vector `[rvec(3), tvec(3), focal_px, cx, cy, k1, k2]`; build `K_j` from `[focal_px, cx, cy]` and ignore distortion terms `[k1, k2]` in the first-pass linear solver.
- `observations`: the bootstrap observation map returned by `_prepare_observations_for_bootstrap(...)`, i.e. `{fid: {cid: (uvA, uvB)}}`.
- `plane_n`: the fixed current normal seed for the window.
- `window_media`: needed only for `n1/n2/n3` and `thickness`.
- `cam_to_window` and `active_cam_ids`: used to filter cameras to the current window and active set.
- Return a dict containing at least:
  - `d_solved`
  - `plane_pt_solved`
  - `accepted`
  - `fallback_reason`
  - `A_shape`, `rank`, `cond`
  - `n_equations`, `n_pairs_used`
  - `residual_rms`
  - `camera_side_ok`

### Data grouping rule
- The solver must treat the two wand endpoints independently:
  - for each frame `fid`, process `uvA` and `uvB` separately
  - each endpoint contributes constraints only from cameras that observe that same endpoint
- For each endpoint, build equations from **all camera pairs** `(j, k)` with valid observations.
- Filter cameras for a window exactly the same way as the current initializer: use cameras where `cam_to_window[cid] == wid` and, if provided, `cid in active_cam_ids`.

### Refracted-ray construction rule
- For each camera `j`, compute the unit incoming ray from the pixel using the bootstrap intrinsics/extrinsics:
  - `u_cam = normalize(K_j^{-1} [u, v, 1]^T)`
  - `u_ij = normalize(R_j^T * u_cam)`
  - `C_j = camera_center(R_j, t_j)`
- Use the same refractive-index order already stored in `window_media` (`n1`, `n2`, `n3`) and the fixed `plane_n` to compute the object-side refracted direction `v_ij`.
- If the native Python vector-Snell helper already exists in repo utilities, reuse it; otherwise implement the standard vector Snell form directly in the solver.
- Explicit one-interface refraction formula to use for each interface transition `n_a -> n_b`:
  
  `eta = n_a / n_b`
  
  `c = -dot(n_hat, u)`
  
  `k = 1 - eta^2 * (1 - c^2)`
  
  if `k < 0`, mark the camera pair as total-internal-reflection and skip it
  
  otherwise the transmitted direction is
  
  `t = eta * u + (eta * c - sqrt(k)) * n_hat`
  
  where `n_hat` is the unit interface normal used by the Snell formula.
- Use `n_hat = -plane_n` for both interfaces so the normal points toward the incoming-ray side under the repo convention where `plane_n` points camera-side -> object-side. Verify `c = -dot(n_hat, u) > 0`; otherwise reject that observation.
- For the two-interface plate model:
  - compute first-interface transmitted direction `t1`
  - propagate inside the plate to the second interface via `Q_exit = Q_ij + (h / (plane_n^T t1)) * t1`
  - apply the same Snell formula again at the second interface to obtain final object-side direction `v_ij`
- `v_ij` depends only on `u_ij`, `plane_n`, and `(n1, n2, n3)`, not on `d`, so `c_i = v_ij × v_ik` is constant with respect to `d` and the solve remains exactly linear.
- Reject total-internal-reflection cases for the equation assembly of that camera pair.

### Equation assembly rule
- For each valid camera pair `(j, k)` observing the same endpoint:
  - compute `Q_ij0`, `q_ij`, `Q_ik0`, `q_ik`
  - compute `c_i = v_ij × v_ik`
  - if `||c_i|| / (||v_ij|| * ||v_ik||) < sin(1 degree)`, skip the pair as near-degenerate
  - add `A_i = (q_ij - q_ik) · c_i` and `b_i = -((Q_ij0 - Q_ik0) · c_i)`
- Stack all `A_i` and `b_i` into a single linear solve.

### Output-to-plane conversion rule
- Use the same anchor as BA for the target window: `A = mean(camera_centers_for_window)`.
- Convert the solved scalar to plane point as `plane_pt_solved = A + d_solved * plane_n`.
- Log both `d_solved` and `plane_pt_solved` for the window.

### Must NOT Have (guardrails, AI slop patterns, scope boundaries)
- Do **not** redesign BA or move this solve into the alternating loop.
- Do **not** change `RefractiveBAOptimizer` parameterization, `_plane_anchor`, or `_plane_d0` logic.
- Do **not** change C++ code, pybind bindings, bootstrap pair selection, or cam-file export semantics.
- Do **not** optimize or refine thickness in the new initializer.
- Do **not** require manual verification; all checks must be agent-executable.

## Verification Strategy
> ZERO HUMAN INTERVENTION - all verification is agent-executed.
- Test decision: tests-after using `pytest` in the `OpenLPT` conda environment
- QA policy: Every task includes agent-executed scenarios
- Evidence: `.sisyphus/evidence/task-{N}-{slug}.{ext}`
- Primary verification source for `case_011`: generated log from `run_calibration_worker.run_one_case(...)`
- Baseline acceptance targets from current `case_011` run:
  - final ray RMSE / `ray_mean_mm` stays **≤ 0.002 mm**
  - final wand-length RMSE / `len_mean_mm` stays **≤ 0.005 mm**
  - all final plane-side checks pass
- Initialization-specific acceptance targets:
  - each initialized window logs `[PLANE_INIT]` winner diagnostics including `d_solved`
  - Stage-0 `dot(n, plane_pt - C_mean) > 0` remains true
  - no active camera violates `s(C) < 0`
  - Stage-0 acceptance is based on valid analytical-`d` diagnostics plus passing side checks; direct `d_gt` comparison is optional and only allowed if the executor explicitly derives the synthetic-frame transform

## Execution Strategy
### Parallel Execution Waves
> Target: 5-8 tasks per wave. <3 per wave (except final) = under-splitting.
> Extract shared dependencies as Wave-1 tasks for max parallelism.

Wave 1: analytical `d` solver, synthetic tests, and fallback foundations
Wave 2: Stage-0 integration, seam regression tests, and `case_011` verification

### Dependency Matrix (full, all tasks)
- Task 1 blocks Tasks 2-4
- Task 2 blocks Tasks 4-5
- Task 3 blocks Tasks 4-5
- Task 4 blocks Tasks 5-8
- Task 5 blocks Tasks 7-8
- Task 6 blocks Task 8
- Task 7 can run after Task 5
- Task 8 depends on Tasks 4-7

### Agent Dispatch Summary (wave → task count → categories)
- Wave 1 → 4 tasks → `quick`, `unspecified-low`
- Wave 2 → 4 tasks → `unspecified-low`, `deep`

## TODOs
> Implementation + Test = ONE task. Never separate.
> EVERY task MUST have: Agent Profile + Parallelization + QA Scenarios.

- [ ] 1. Add a pure analytical solver for fixed-normal plane offset `d` from 2D correspondences

  **What to do**: Create `modules/camera_calibration/wand_calibration/plane_d_solver.py` (or place an equivalently scoped function in an existing geometry module) with pure, testable helpers that do not mutate BA state. Implement `solve_plane_d_from_correspondences(...)` for a fixed `plane_n`, known thickness, and multi-view 2D endpoint correspondences. The solver must: (a) compute incident rays `u_ij` from `K`, `rvec`, and `tvec`; (b) compute interface intersections `Q_ij(d)` analytically; (c) apply Python Snell/refraction steps for the two-interface plate model; (d) assemble one scalar linear equation `A_i d = b_i` per valid multi-view correspondence; and (e) solve `A d = b` with least squares to produce `d_solved`, diagnostics, and validity flags.
  **Must NOT do**: Do not touch `RefractiveBAOptimizer`, C++ camera code, or any export path. Do not change or re-estimate `plane_n` in this plan. Do not refine thickness. Do not read external case files inside the solver. Do not accept a solve when fewer than 2 cameras observe a correspondence or when the linear system is rank-deficient / ill-conditioned.

  **Recommended Agent Profile**:
  - Category: `quick` - Reason: isolated analytical solver with local math utilities and no architecture changes
  - Skills: `[]` - no special skill required beyond repo-local pattern following
  - Omitted: `['test-driven-development']` - not required as a skill injection because the plan already prescribes explicit test-first steps

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: [2, 3, 4] | Blocked By: []

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `modules/camera_calibration/wand_calibration/refraction_wand_calibrator.py:455-492` - current camera-center aggregation and legacy `d0_mm` / `plane_pt` construction to replace
  - Pattern: `modules/camera_calibration/wand_calibration/refraction_wand_calibrator.py:505-553` - sign-disambiguation and orientation scoring contract that must remain compatible after replacing `d`
  - API/Type: `modules/camera_calibration/wand_calibration/refraction_wand_calibrator.py:403-413` - current initializer signature and data inputs
  - API/Type: `modules/camera_calibration/wand_calibration/refraction_calibration_BA.py:1222-1234` - downstream anchor parameterization `pt = A + d * n` that the Stage-0 seed must stay compatible with
  - Test: `tests/test_plane_pipeline.py:624-679` - initializer seam is already treated as swappable and can be monkeypatched in tests
  - External: `https://openaccess.thecvf.com/content_cvpr_2014/html/Chen_Two-View_Camera_Housing_2014_CVPR_paper.html` - correspondence-based literature inspiration; use only as motivation, not as a direct algorithm port

  **Acceptance Criteria** (agent-executable only):
  - [ ] `modules/camera_calibration/wand_calibration/plane_d_solver.py` exists and exposes a pure `solve_plane_d_from_correspondences(...)` API.
  - [ ] `conda run -n OpenLPT python -m pytest tests/test_plane_d_solver.py -v -k synthetic` passes.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: Synthetic 2-camera solve recovers plane offset
    Tool: Bash
    Steps: Run `conda run -n OpenLPT python -m pytest tests/test_plane_d_solver.py -v -k synthetic_two_cam`
    Expected: PASS; solver recovers `d` within the threshold defined by the test fixture
    Evidence: .sisyphus/evidence/task-1-solver.txt

  Scenario: Rank-deficient system falls back safely
    Tool: Bash
    Steps: Run `conda run -n OpenLPT python -m pytest tests/test_plane_d_solver.py -v -k degenerate`
    Expected: PASS; solver reports invalid/ill-conditioned solve and exposes fallback-ready diagnostics
    Evidence: .sisyphus/evidence/task-1-solver-error.txt
  ```

  **Commit**: YES | Message: `feat(wand-cal): add analytical plane-d solver` | Files: [`modules/camera_calibration/wand_calibration/plane_d_solver.py`, `tests/test_plane_d_solver.py`]

- [ ] 2. Add synthetic unit tests for `d` recovery accuracy, conditioning, and fallback conditions

  **What to do**: Create `tests/test_plane_d_solver.py` with explicit synthetic fixtures covering (a) 2-camera known-plane recovery, (b) 3+ camera overdetermined recovery, (c) near-parallel / ill-conditioned rays, and (d) insufficient multi-view overlap. Generate 2D correspondences from known synthetic geometry, run `solve_plane_d_from_correspondences(...)`, and verify returned `d` plus validity diagnostics.
  **Must NOT do**: Do not instantiate full calibration runs here. Do not depend on `J:` assets in these unit tests. Do not assert only qualitative improvement; assert explicit numerical `d` error thresholds.

  **Recommended Agent Profile**:
  - Category: `quick` - Reason: isolated Python tests around one new analytical solver
  - Skills: `[]` - local test authoring only
  - Omitted: `['verification-before-completion']` - final verification is covered later at integration level

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: [4] | Blocked By: [1]

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `tests/test_plane_pipeline.py:624-679` - pytest style, numpy assertions, monkeypatch patterns
  - API/Type: `modules/camera_calibration/wand_calibration/plane_d_solver.py` - pure solver API from Task 1
  - Math Contract: each valid correspondence contributes one scalar equation `A_i d = b_i`; solve with `np.linalg.lstsq`

  **Acceptance Criteria** (agent-executable only):
  - [ ] Unit tests cover positive and negative `d`-solver cases without relying on external datasets.
  - [ ] `conda run -n OpenLPT python -m pytest tests/test_plane_d_solver.py -v -k synthetic` passes.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: Overdetermined multi-camera solve is accurate
    Tool: Bash
    Steps: Run `conda run -n OpenLPT python -m pytest tests/test_plane_d_solver.py -v -k overdetermined`
    Expected: PASS; 3+ camera fixture recovers `d` within the stricter threshold defined in the test
    Evidence: .sisyphus/evidence/task-2-unit.txt

  Scenario: Ill-conditioned geometry triggers fallback-ready outputs
    Tool: Bash
    Steps: Run `conda run -n OpenLPT python -m pytest tests/test_plane_d_solver.py -v -k ill_conditioned`
    Expected: PASS; tests verify fallback-ready status instead of unstable `d` estimates
    Evidence: .sisyphus/evidence/task-2-unit-error.txt
  ```

  **Commit**: YES | Message: `test(wand-cal): cover analytical plane-d solver` | Files: [`tests/test_plane_d_solver.py`]

- [ ] 3. Add solver validation and fallback gating for analytical `d` vs legacy midpoint depth

  **What to do**: Extend `plane_d_solver.py` with validation and fallback gating utilities that compare the analytical `d_solved` against the legacy midpoint-depth seed. The gate must preserve the current `plane_n`, compute `plane_pt = A + d * plane_n`, and reject the analytical result if any active camera violates the camera-side rule, if the linear system is ill-conditioned, or if `d_solved` is non-finite / physically implausible. Keep the legacy midpoint-depth seed as the fallback.
  **Must NOT do**: Do not call `RefractiveBAOptimizer.optimize()`. Do not require changing `RefractiveBAConfig`. Do not use a full external harness here.

  **Recommended Agent Profile**:
  - Category: `unspecified-low` - Reason: local algorithmic utility with some design care but limited surface area
  - Skills: `[]` - local code reasoning only
  - Omitted: `['systematic-debugging']` - this is planned feature work, not a reactive bug hunt

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: [4, 5] | Blocked By: [1]

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `modules/camera_calibration/wand_calibration/refraction_wand_calibrator.py:485-497` - depth heuristic and `plane_pt` construction that must stay unchanged
  - Pattern: `modules/camera_calibration/wand_calibration/refraction_wand_calibrator.py:505-523` - current orientation/object-side scoring structure
  - Pattern: `modules/camera_calibration/wand_calibration/refraction_wand_calibrator.py:555-578` - Stage-0 sanity outputs that must remain satisfiable
  - API/Type: `modules/camera_calibration/wand_calibration/refraction_calibration_BA.py:417-443` - BA later rebuilds anchor-relative distance; Stage-0 scorer should not change this contract
  - API/Type: `modules/camera_calibration/wand_calibration/refractive_geometry.py:1233-1256` - signed-depth / camera-side constraints the solved plane must satisfy

  **Acceptance Criteria** (agent-executable only):
  - [ ] Validation utilities return comparable diagnostics for analytical `d` and legacy midpoint depth while leaving `plane_n` unchanged.
  - [ ] Candidate acceptance requires all active cameras to remain camera-side, finite solved `d`, and a non-degenerate linear solve; otherwise legacy wins.
  - [ ] `conda run -n OpenLPT python -m pytest tests/test_plane_d_solver.py -v -k fallback` passes.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: Analytical solve and legacy fallback are ranked deterministically
    Tool: Bash
    Steps: Run `conda run -n OpenLPT python -m pytest tests/test_plane_d_solver.py -v -k fallback`
    Expected: PASS; repeated runs over the same fixtures choose the same accepted or fallback path and expose structured diagnostics
    Evidence: .sisyphus/evidence/task-3-scoring.txt

  Scenario: Invalid analytical solve preserves legacy midpoint depth
    Tool: Bash
    Steps: Run the same fallback test subset with fixtures forcing ill-conditioned solves
    Expected: PASS; tests verify fallback uses the legacy `d0_mm = median_depth / n_object` path
    Evidence: .sisyphus/evidence/task-3-scoring-error.txt
  ```

  **Commit**: YES | Message: `feat(wand-cal): add plane-d fallback gating` | Files: [`modules/camera_calibration/wand_calibration/plane_d_solver.py`, `tests/test_plane_d_solver.py`]

- [ ] 4. Integrate branch-and-compare selection into the existing Stage-0 plane initializer seam

  **What to do**: Modify `modules/camera_calibration/wand_calibration/refraction_wand_calibrator.py` so `PlaneInitializer.init_window_planes_from_cameras(...)` keeps the current normal heuristic, replaces the legacy `d0_mm = median_depth / n_object` seed with `solve_plane_d_from_correspondences(...)`, and falls back to the legacy midpoint-depth seed when the analytical solve is invalid. Preserve the existing function signature and returned structure. Add deterministic logs of the form `[PLANE_INIT] Win {wid}: d_midpoint=... d_solved=... chose=... fallback_reason=...`.
  **Must NOT do**: Do not move initialization into BA. Do not change `_init_cams_cpp_in_memory`, `RefractiveBAOptimizer`, or any export logic. Do not remove the current midpoint path; it is the fallback.

  **Recommended Agent Profile**:
  - Category: `unspecified-low` - Reason: existing orchestration integration with contract preservation requirements
  - Skills: `[]` - repo-local integration only
  - Omitted: `['using-git-worktrees']` - workspace rule forbids separate worktree use unless explicitly requested

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: [5, 6, 7, 8] | Blocked By: [1, 2, 3]

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `modules/camera_calibration/wand_calibration/refraction_wand_calibrator.py:403-592` - full current initializer body to preserve behavior and logs
  - Pattern: `modules/camera_calibration/wand_calibration/refraction_wand_calibrator.py:2208-2213` - Stage-0 call site that uses this seam before BA
  - Pattern: `modules/camera_calibration/wand_calibration/refraction_wand_calibrator.py:2241-2279` - downstream camera-side sanity checks that must continue to pass
  - API/Type: `modules/camera_calibration/wand_calibration/refraction_calibration_BA.py:421-443` - downstream BA expects Stage-0 `plane_pt` and `plane_n` only; no other contract changes allowed
  - Test: `tests/test_plane_pipeline.py:624-679` - seam already treated as live initializer source in trace tests

  **Acceptance Criteria** (agent-executable only):
  - [ ] `PlaneInitializer.init_window_planes_from_cameras(...)` remains callable with the same signature and returns the same key structure.
  - [ ] The initializer emits `[PLANE_INIT]` winner diagnostics for each initialized window including both `d_midpoint` and `d_solved` when available.
  - [ ] `conda run -n OpenLPT python -m pytest tests/test_plane_d_solver.py tests/test_plane_pipeline.py -v -k integration` passes.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: Stage-0 selects and logs a winning candidate
    Tool: Bash
    Steps: Run `conda run -n OpenLPT python -m pytest tests/test_plane_d_solver.py tests/test_plane_pipeline.py -v -k integration`
    Expected: PASS; tests verify selected plane location matches expected winner and logs include `[PLANE_INIT]`
    Evidence: .sisyphus/evidence/task-4-integration.txt

  Scenario: Weak geometry triggers automatic midpoint fallback
    Tool: Bash
    Steps: Run the same integration subset with fixtures for insufficient overlap / ill-conditioned solves
    Expected: PASS; tests verify silent fallback to the legacy midpoint path and a populated fallback reason
    Evidence: .sisyphus/evidence/task-4-integration-error.txt
  ```

  **Commit**: YES | Message: `feat(wand-cal): integrate analytical plane-d init` | Files: [`modules/camera_calibration/wand_calibration/refraction_wand_calibrator.py`, `modules/camera_calibration/wand_calibration/plane_d_solver.py`, `tests/test_plane_d_solver.py`, `tests/test_plane_pipeline.py`]

- [ ] 5. Add seam regression coverage in `tests/test_plane_pipeline.py` for default-on selection and fallback metadata

  **What to do**: Extend `tests/test_plane_pipeline.py` with focused tests that confirm the Stage-0 trace continues to treat the live initializer as authoritative, and that the new Stage-0 metadata includes solved `d`, legacy `d_midpoint`, selected source, and fallback reason when the analytical solver wins or falls back. Reuse the existing monkeypatch seam rather than invoking the full case harness.
  **Must NOT do**: Do not rewrite unrelated trace tests. Do not hard-code `J:` case dependencies into this file. Do not weaken existing assertions about `INIT` source and live initializer output.

  **Recommended Agent Profile**:
  - Category: `quick` - Reason: targeted regression tests in an existing test file
  - Skills: `[]` - pytest and monkeypatch only
  - Omitted: `['requesting-code-review']` - not needed at task granularity

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: [8] | Blocked By: [4]

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `tests/test_plane_pipeline.py:545-559` - monkeypatch injection point for `PlaneInitializer.init_window_planes_from_cameras`
  - Pattern: `tests/test_plane_pipeline.py:624-679` - live init stage source, metadata, and camera-center override assertions
  - Pattern: `tests/test_plane_pipeline.py:682-694` - trace transform assertions that must not regress when init metadata grows

  **Acceptance Criteria** (agent-executable only):
  - [ ] Existing trace tests still pass or are updated only where new metadata is intentionally added.
  - [ ] New seam regression tests verify winner/fallback metadata without requiring external assets.
  - [ ] `conda run -n OpenLPT python -m pytest tests/test_plane_pipeline.py -v -k correspondence` passes.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: Trace captures correspondence-init metadata
    Tool: Bash
    Steps: Run `conda run -n OpenLPT python -m pytest tests/test_plane_pipeline.py -v -k correspondence`
    Expected: PASS; INIT stage artifact still points to live initializer output and carries `d_solved` / fallback metadata
    Evidence: .sisyphus/evidence/task-5-trace.txt

  Scenario: Existing live-init source assertions remain intact
    Tool: Bash
    Steps: Run the same subset and include prior live-init tests touched by metadata updates
    Expected: PASS; no regression in source=`live_init_window_planes_from_cameras` expectations
    Evidence: .sisyphus/evidence/task-5-trace-error.txt
  ```

  **Commit**: YES | Message: `test(wand-cal): extend plane trace init metadata` | Files: [`tests/test_plane_pipeline.py`]

- [ ] 6. Add a `case_011` verification helper test that runs the synthetic harness through `run_one_case(...)`

  **What to do**: Add an end-to-end integration test that imports `run_one_case` from `J:\Refraction_test\test_script\run_calibration_worker.py`, runs `case_011`, captures the returned metrics plus generated log path, and asserts the non-regression thresholds and Stage-0 diagnostics required by this plan. Parse the generated log for `[PLANE_INIT]` diagnostics and any frame-count / side-check summaries needed by the assertions. Guard the test with `pytest.skip` when the external harness or case directory is unavailable.
  **Must NOT do**: Do not modify `J:` harness files. Do not depend on batch-only `run_robustness.py` for single-case execution. Do not use GT pose error metrics (`r_err_mean_deg`) as the primary acceptance signal because the existing baseline is inconsistent there. Do not require direct `d_gt` comparison unless the executor also adds an explicit frame-transform derivation from case metadata into the bootstrap-aligned frame.

  **Recommended Agent Profile**:
  - Category: `unspecified-low` - Reason: external-harness integration with careful skip logic
  - Skills: `[]` - local test code only
  - Omitted: `['webapp-testing']` - not a browser workflow

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: [8] | Blocked By: [4]

  **References** (executor has NO interview context - be exhaustive):
  - External: `J:/Refraction_test/test_script/run_calibration_worker.py` - use `run_one_case(case_dir, runtime_root, smoke_n_frames=0, verbosity=1)` as the single-case entrypoint
  - External: `J:/Refraction_test/case_011/case_meta.json` - case geometry: 2 windows, 5 cameras, 2000 frames
  - External: `J:/Refraction_test/test_results/logs/robustness/case_011.log` - baseline Stage-0 and final diagnostics source
  - Pattern: `tests/test_plane_pipeline.py` - existing style for optional external-asset tests with guardable setup

  **Acceptance Criteria** (agent-executable only):
  - [ ] Test skips cleanly when `J:\Refraction_test\case_011` or `run_calibration_worker.py` is missing.
  - [ ] When available, the test asserts `success=True`, `ray_mean_mm <= 0.002`, `len_mean_mm <= 0.005`, presence of `[PLANE_INIT]` in the log, and passing Stage-0 side-check diagnostics parsed from the log.
  - [ ] `conda run -n OpenLPT python -m pytest tests/test_plane_d_solver.py -v -k case011` passes.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: case_011 passes non-regression thresholds
    Tool: Bash
    Steps: Run `conda run -n OpenLPT python -m pytest tests/test_plane_d_solver.py -v -k case011`
    Expected: PASS or SKIP with explicit environment reason; when PASS, metrics satisfy thresholds, log contains `[PLANE_INIT]`, and parsed Stage-0 side checks pass
    Evidence: .sisyphus/evidence/task-6-case011.txt

  Scenario: Missing external assets skip cleanly
    Tool: Bash
    Steps: Run the same test in an environment without `J:` assets or simulate missing path in the test fixture
    Expected: PASS; test reports SKIP with actionable reason instead of erroring
    Evidence: .sisyphus/evidence/task-6-case011-error.txt
  ```

  **Commit**: YES | Message: `test(wand-cal): add case011 plane-d regression` | Files: [`tests/test_plane_d_solver.py`]

- [ ] 7. Add a forced-legacy comparison path for automated non-regression against the old initializer

  **What to do**: Introduce a minimal, local control surface that allows tests to run the legacy midpoint-only initializer path explicitly so the new default-on behavior can be compared against the old baseline in automation. Implement this as an internal environment override `OPENLPT_REFRACTION_PLANE_INIT_MODE` with allowed values `auto` (default) and `legacy`, read inside `PlaneInitializer.init_window_planes_from_cameras(...)`. Production behavior must remain `auto`.
  **Must NOT do**: Do not make legacy mode the default. Do not require users to toggle it in normal operation. Do not expose a sprawling public API surface just for the test.

  **Recommended Agent Profile**:
  - Category: `unspecified-low` - Reason: limited control-surface addition with regression-testing purpose
  - Skills: `[]` - local API/threading only
  - Omitted: `['brainstorming']` - design decision already made in this plan

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: [8] | Blocked By: [4]

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `modules/camera_calibration/wand_calibration/refraction_wand_calibrator.py:2208-2213` - Stage-0 seam that can accept an internal mode switch without touching BA
  - Pattern: `modules/camera_calibration/wand_calibration/refraction_wand_calibrator.py:403-592` - legacy midpoint path remains the fallback and comparison baseline
  - Test: `tests/test_plane_d_solver.py` - use this file to compare default-on vs forced-legacy results

  **Acceptance Criteria** (agent-executable only):
  - [ ] Tests can force the legacy midpoint-only path without changing default production behavior.
  - [ ] `conda run -n OpenLPT python -m pytest tests/test_plane_d_solver.py -v -k legacy_compare` passes.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: Default-on and forced-legacy paths are both runnable in tests
    Tool: Bash
    Steps: Run `conda run -n OpenLPT python -m pytest tests/test_plane_d_solver.py -v -k legacy_compare`
    Expected: PASS; tests exercise both paths and confirm production default remains correspondence+fallback
    Evidence: .sisyphus/evidence/task-7-legacy.txt

  Scenario: Forced-legacy control does not leak into normal execution
    Tool: Bash
    Steps: Run the same subset and assert default code path still logs `[PLANE_INIT]` winner diagnostics while forced legacy does not alter unrelated config
    Expected: PASS; control surface remains scoped and non-invasive
    Evidence: .sisyphus/evidence/task-7-legacy-error.txt
  ```

  **Commit**: YES | Message: `test(wand-cal): add legacy plane-d comparison path` | Files: [`modules/camera_calibration/wand_calibration/refraction_wand_calibrator.py`, `tests/test_plane_d_solver.py`]

- [ ] 8. Run the full verification set and record baseline-vs-new acceptance evidence

  **What to do**: Execute the planned verification commands in the `OpenLPT` conda environment, capture outputs, and confirm the implementation meets all acceptance thresholds. Review generated logs for `case_011` to confirm `[PLANE_INIT]`, Stage-0 camera/object-side checks, and non-regressed final ray/length metrics. If any threshold fails, fix the implementation before completion.
  **Must NOT do**: Do not claim success without command output. Do not use `python` from an unknown environment; use `conda run -n OpenLPT python ...` per repo rules. Do not skip the `case_011` log inspection.

  **Recommended Agent Profile**:
  - Category: `deep` - Reason: this is the final evidence-gathering and non-regression checkpoint across unit, seam, and external-harness coverage
  - Skills: `[]` - command execution and evidence review only
  - Omitted: `['proactive-verify']` - verification is explicitly scripted in this task already

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: [Final Verification Wave] | Blocked By: [4, 5, 6, 7]

  **References** (executor has NO interview context - be exhaustive):
  - Command: `conda run -n OpenLPT python -m pytest tests/test_plane_d_solver.py -v`
  - Command: `conda run -n OpenLPT python -m pytest tests/test_plane_pipeline.py -v -k correspondence`
  - Command: `conda run -n OpenLPT python -c "import sys; sys.path.insert(0, r'J:/Refraction_test/test_script'); from run_calibration_worker import run_one_case; result = run_one_case(r'J:/Refraction_test/case_011', r'J:/Refraction_test/test_results'); print(result['success']); print(result['metrics'])"`
  - External: `J:/Refraction_test/test_results/logs/robustness/case_011.log` - inspect for `[PLANE_INIT]`, `WIN_SANITY`, and final diagnostics

  **Acceptance Criteria** (agent-executable only):
  - [ ] All planned pytest commands pass.
  - [ ] `run_one_case(...)` returns `success=True`.
  - [ ] `ray_mean_mm <= 0.002` and `len_mean_mm <= 0.005`.
  - [ ] `case_011.log` contains `[PLANE_INIT]` plus passing Stage-0 and final plane-side diagnostics.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: Full verification suite passes
    Tool: Bash
    Steps: Run the three verification commands listed above in the OpenLPT conda environment
    Expected: PASS; all tests succeed and `run_one_case` prints success with non-regressed metrics
    Evidence: .sisyphus/evidence/task-8-verify.txt

  Scenario: Threshold regression is caught before completion
    Tool: Bash
    Steps: Re-run the case_011 harness after any implementation fix and inspect the log/metrics for threshold violations
    Expected: PASS; any regression blocks completion until fixed, and final evidence shows compliant thresholds
    Evidence: .sisyphus/evidence/task-8-verify-error.txt
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
- Commit 1: `test(wand-cal): add analytical plane-d solver tests`
- Commit 2: `feat(wand-cal): add analytical plane-d solver`
- Commit 3: `feat(wand-cal): integrate stage0 plane-d init fallback`
- Commit 4: `test(wand-cal): add case011 plane-d verification`

## Success Criteria
- New Stage-0 initializer runs by default and preserves legacy behavior via fallback.
- Existing seam and trace tests keep passing.
- `case_011` remains non-regressed on final metrics and logs the new Stage-0 winner diagnostics.
- No downstream BA, C++, bootstrap, or export contracts are broken.
