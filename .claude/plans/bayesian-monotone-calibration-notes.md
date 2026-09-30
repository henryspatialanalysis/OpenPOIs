# Running notes: Bayesian monotone-spline calibration (Phase 1)

Execution log for `.claude/plans/bayesian-monotone-calibration.md`, the design doc with
Nat's decisions in §9. The approved execution plan is
`~/.claude/plans/pleask-ask-me-any-precious-dream.md`. Every decision or assumption the
plan left open is recorded here with its reason; newest entries go at the bottom of
each section.

## Setup

- **Branch.** Work continues on `feature/matched-index-modes`, the branch that was
  checked out when the plan was approved. Nothing is committed until Nat's commit
  session.
- **Validation round.** `data/calibration/20260730/`: 7,504 rows, 2,362 gold. The 4
  `overture_missing_conf` rows are excluded, as the production fit excludes them.
- **Output directory:** `~/data/openpois/conflation/20260730/calibration_eval_bayes_20260927/`.

## Decisions

1. **Step 0 done (2026-09-27).** The design doc now matches decisions 1–11:
   - the nugget is removed from M2, M4, M16 and M18, and from §3.7, §3.8, §4 and §5;
   - §6.1 is the rationale for dropping it;
   - arm C is §3.5b (M15b–d);
   - §9 is the decisions table;
   - citations are author–year throughout, and §10 is one alphabetical list in full
     format. Kumar et al. (2019) was dropped as unused, and Cox (1993), Deng et al.
     (2021), Kohavi (1995) and the mbg vignette were added.
2. **The `jax_core` passthrough** is `adaptation_kwargs: dict | None = None` on
   `_nuts_sample_core`, `nuts_sample` and `nuts_sample_multichain`, forwarded to
   `blackjax.window_adaptation`.
   - The default path is unchanged. A test compares it against a direct BlackJAX run
     with the same keys.
3. **Code review** (a `pr-code-reviewer` pass, 2026-09-27) found no breaking bugs.
   All eight findings are adopted:
   - Missing scores and gold `y` now raise in `prepare_data`, where they used to be
     silently filled. `curve_draws` returns NaN for missing points.
   - CV folds are assigned **per segment** with a fresh RNG
     (`assign_segment_folds`). Calling the helper once on the full table would have
     put 2,116 of 2,361 gold rows in a different fold from production. A test now
     checks all three segments against `cross_fit_predictions`.
   - Tree-depth saturation is `num_integration_steps >= 1023` (Stan's definition), not
     BlackJAX's 10th-expansion counter.
   - Arm C raises if a non-gold, non-held-out row has no silver label (selection on
     v).
   - `find_map` falls back to the initial values on a non-finite result.
   - `fit` warns when x64 is off.
   - New tests cover the tensor-product orientation against scipy on an asymmetric
     matrix, arm A recovery, and the `jax_core` default path.
4. **The level anchor moves to the data centre** (a reparametrisation, not a model
   change).
   - M7 and M9a pin α at the first coefficient (score 0, or the matched corner),
     where there are almost no data. α is then strongly correlated with the
     slopes.
   - `ModelSpec.anchor = "center"` (default) pins α at the coefficient nearest the
     segment's median phase-1 score (for matched, the median of each axis).
   - Both the cumulative sum and the max recursion are shift-equivariant, so this is
     the same family of curves.
   - The prior N(0, 1.5²) now applies at the data centre, where it is still weakly
     informative. **Methodological note for Nat.**
5. **Log-slope cap.** γ is capped at 25 before exponentiation. That is ~7e10 per
   unit score, far outside the prior. It only matters in fp32, where an overflowed
   step with a mid-curve anchor gives inf − inf = NaN; the fp32 unit test found this.
6. **Speed.** The smoke fit (arm A, 2 chains × 200 iterations) took 7 min at about 250
   leapfrog steps per iteration, even though a gradient costs only 2.5 ms. The cost
   is geometry. Dense vs diagonal mass matrices and the centre anchor are being
   benchmarked (logs `bench_*.log`) before the full fits are launched.
7. **Measurement-layer parameterisation** (sampler benchmarks, 2026-09-28; logs
   `bench*.log`).

   | run (2 chains × 600 iterations) | minutes | steps/iteration | max R̂ | min ESS |
   |---|---|---|---|---|
   | diagonal mass, origin anchor, non-centred M13 | 27 | 127 | 1.06 | 40 |
   | diagonal mass, centre anchor, non-centred M13 | 31 | 206 | 1.05 | 56 |
   | dense mass (origin anchor) | 67 | 1023 (saturated) | 1.70 | 3 |
   | centred M13, origin anchor (800 iterations) | 46 | 177 | 1.12 | 14 |
   | centred M13, centre anchor (800 iterations) | 47 | 134 | 1.20 | 9 |

   - The dense mass matrix is ruled out.
   - With centring, every curve parameter mixes well (ESS 200 or more). The
     remaining slow block is the y = 0 (truly gone) class logits in the Overture
     segment. Under the reference-class softmax of M12, all of them are measured
     against exists:high, which is rare given y = 0. They therefore share one
     weakly identified offset, a ridge.
   - **Change: the softmax uses a sum-to-zero orthonormal (Helmert) basis**
     (`softmax_basis = "sum_to_zero"`) instead of pinning exists:high at 0. It is the
     same likelihood family, with an exchangeable prior over classes.
   - **Change: M13 is centred** (`centered_measurement = True`): ψ ~ N(μ_ψ, ω_ψ²)
     sampled directly. Each segment's ~2,500 rows pin ψ_g, which is the regime where
     centring beats non-centring.
   - Both are **methodological notes for Nat**. The prior on the measurement logits
     is now symmetric across classes, not "log-odds against exists:high".
8. **Sampler settings.**
   - Main full-data fits: 4 chains × (1,000 warmup + 1,000 draws).
   - CV folds, coverage rounds and sensitivity runs: 4 chains × (600 + 400). They need
     posterior means and 95% bands, not tail-quantile precision, and the full setting
     would take ~15 h of CPU for about 80 fits.
   - The §5.1 acceptance rule is applied as written to the main fits. The light fits'
     diagnostics are reported beside their results.

9. **Smooth max in the 2-D recursion** (§5.1 escalation, 2026-09-28).
   - The arm B main fit with the exact max had 216 divergences in 4,000 draws
     (acceptance 0.71). The slowest parameters were τ_M and z_matched.
   - Escalation step 1, target acceptance 0.95: worse (ESS 71, 281 steps, still 31
     divergences in 1,600).
   - Step 2, dense mass matrix: catastrophic in the benchmark (tree-depth
     saturation, R̂ 1.7).
   - Step 3, smooth max with t = 0.05: divergences 28/1,600 (from 5.4% to 1.8%),
     3× faster.
   - **Adopted t = 0.05 for every arm** (`ModelSpec.smooth_max_t` default). The
     forced excess per cell is at most 0.05·log 2 ≈ 0.035 on the c~ scale. Monotonicity
     is unchanged.
   - The exact-max arm B and C fits are kept as `fits/armB_exactmax` and
     `fits/armC_exactmax`. Arm C exact-max was clean: R̂ 1.005, ESS 1,351, 1
     divergence. A divergence-location diagnosis for arm B is in
     `logs/diag_div_B.log`.
10. **Arm B keeps some divergences** (accepted, reported).
    - Divergence-location diagnosis (`logs/diag_div_B.log`, 39 of 1,600): the
      divergent draws sit at large τ (τ_overture ≈ 1.5) and at extreme values of the
      smoothest matched eigen-modes. That is the heavy tail of exp(log-slope): very
      steep curves saturate the squash.
    - The gold-only pseudo-likelihood has the least data to rule those out.
    - The smooth-max arm B main fit had 150 divergences in 4,000 (R̂ 1.010, ESS bulk
      699, tail 292).
    - Arm B is a point-prediction comparator only, so this is tolerated and flagged
      rather than fixed by tightening the τ prior for one arm.
11. **Arm A structure tests.** Arm A's main fit (2026-09-28, 4 × 1,000 + 1,000)
    failed §5.1:
    - max R̂ 1.09, minimum ESS 35 bulk / 15 tail, 121 divergences (almost all in
      chain 1);
    - the slow block is the OSM segment's y = 0 class logits and slopes; four of the
      nine refined classes are nearly empty given y = 0 there;
    - the PPC verdict mix was only 78% inside its 95% intervals.

    It is also **biased low against the design-weighted gold rate**:
    - matched overall 0.871 vs 0.906, and the lowest raw-score quintile 0.760 vs
      0.863;
    - Overture 0.649 vs 0.663;
    - arms B and C track the gold rate and production (0.908 and 0.916 on matched).

    Likely mechanism: the partial pooling of θ across segments pulls the matched
    segment's P(gone verdict | y = 0) toward the Overture segment's value, where
    truly-gone POIs are mostly "unverifiable". Explaining the matched gone verdicts
    then needs more y = 0, which pushes p down (Little, 2004, on design variables
    modelled with the wrong form).

    Four structure variants were fitted with the light settings:
    - T1: 3 verdict classes;
    - T2: `merged6` (thin confidence levels merged, `MERGE_MAP`);
    - T3: β ≡ 0;
    - T4: unpooled θ (`pool_segments = False`).
12. **Arm C is the main model** (Nat, 2026-09-28).
    - CV arms: C (main), B, and the best arm A variant, which keeps the "does the
      full layer add?" comparison.
    - Sensitivity runs are on arm C. S2 and S3 are replaced by:
      - S2C: asymmetric label noise, with Se and Sp priors centred on the handoff's
        design-weighted Se 0.998 and Sp 0.913 at concentration 64;
      - S3C-a: silver labels exact (β = 1);
      - S3C-b: β ~ Beta(9, 1).
    - The coverage study fits arm C on 20 in-family rounds and 5 rounds each of the
      realistic-LLM (arm A's fitted θ), category and step scenarios. Arm B is fitted
      on the 15 misspecified rounds.
    - Arm C's posterior β is 0.9997: the silver labels are treated as almost exact.
    - Note on arm C's validity:
      - Given the noise model, dropping the gold rows' θ_y(v) factor does not
        change inference on F, because that factor does not involve F.
      - Conditioning non-gold rows on "not unverifiable" ignores the y-dependent
        factor (1 − u_y). That is exact when silver labels are correct and
        approximate otherwise.
      - **Methodological note for Nat.**
13. **Arm A variant results** (light settings; mean P(exists) vs design-weighted gold
    rate; matched segment overall):

    | fit | max R̂ | min ESS bulk / tail | PPC verdict mix inside | matched (gold rate 0.906) | Overture (0.663) | OSM (0.817) |
    |---|---|---|---|---|---|---|
    | armA (refined9, main settings) | 1.09 | 35 / 15 | 78% | 0.871 | 0.649 | 0.807 |
    | T1 verdict3 | 1.15 | 20 / 19 | 81% | — | — | — |
    | T2 merged6 | 1.02 | 375 / 449 | 79% | 0.867 | 0.650 | 0.807 |
    | T3 β ≡ 0 | 1.02 | 248 / 341 | 71% | 0.874 | 0.643 | 0.809 |
    | T4 unpooled | 1.02 | 383 / 191 | — | 0.864 | 0.653 | 0.814 |
    | armC (main) | 1.006 | 1349 / 1053 | label mix 93% | 0.915 | 0.676 | 0.817 |
    | armB | 1.010 | 699 / 292 | — | 0.908 | 0.658 | 0.812 |

    - **No arm A variant removes the bias.** Pooling is not the cause: T4, unpooled,
      is as low as the rest. The measurement layer lets class probabilities depend on
      score only through one linear slope, so it cannot reproduce how the verdict
      mix varies with score (PPC 71–81% in every variant). The curve absorbs the
      misfit, most at low scores (matched lowest quintile 0.755–0.763 vs a gold rate
      of 0.863).
    - T2 (merged6) converges best, so it is the arm A carried into CV as the
      comparator.
    - `fits/armA_variant` is a copy of the light T2 fit. It was not refit at
      1,000 + 1,000, which would take about 3 h for a comparator.
    - Arm A's CV folds use 4 × (400 + 300) (point predictions only).
    - T4 crashed after sampling in `key_parameters` (it assumed the pooled ω_ψ).
      That is fixed, and its `summary.json` is a stub holding the spec.
14. **Arm C diagnostics** (from the saved draws, `--reuse-draws`):
    - PPC: rate by knot interval 90% inside, label mix 93% inside.
    - Deployed impact vs published on the 20260902 population: Overture mean 0.678 vs
      0.683, 12.9% of POIs moving by more than 0.05 (arm A: 57%).
15. **S2C (asymmetric label noise) exposes the limits of arm C's noise model**
    (2026-09-28).
    - Se's posterior is 1.00, but **Sp's collapses to 0.38 [0.34, 0.42]** from a
      prior centred at 0.913 with concentration 64. The curve drops: the rate-by-knot
      PPC is 48% inside, against 90% for arm C.
    - **Mechanism.** At a given score, gold rows are enriched for truly-gone POIs
      (gone verdicts sampled at 37%, unverifiables censused), and non-gold rows for
      "exists". Arm C models both as draws from one p(s). With a free Sp the model
      lowers p to fit the gold rows and explains the non-gold rows' higher "exists"
      rate as gone POIs mislabelled "exists".
    - Symmetric noise cannot do this, because it pulls the silver rate toward 0.5,
      the wrong direction. So β → 1 (posterior mean 0.9997), and **main arm C is in
      effect "silver labels are exact"**. That case is valid: every phase-1 row then
      enters once with its true label (design doc §3.5b).
    - **Implication.** Arm C's noise parameters are not safely learnable from its own
      likelihood. They are identified only through the selection-induced gold/silver
      gap, which is exactly what they should not absorb.
    - A defensible arm C needs its noise pinned. Either take it from the gold
      concordance, adding the gold rows' θ_y(v) factor for definitive verdicts, or fix
      it (S3C-exact).
    - **Methodological question for Nat (Q-c).**
16. **Arm A CV deferred; CV intervals by stratified bootstrap only** (Nat,
    2026-09-28).
    - Arm C over arm A is justified on the full-data evidence (decisions 11–13, 15).
      The arm A CV run was stopped before any fold finished.
    - CV compares arms C and B with production (published lookup and curve).
    - Model-comparison intervals are the paired bootstrap within fold × refined
      class. No Vehtari-style analytic SE.
    - Uncertainty about the curve itself comes from the posterior draws. No
      bootstrap is used anywhere in the Bayesian fitting.
    - Interim CV (arms C and B complete; design-weighted held-out Brier):
      - arm C relative Brier vs production as published: 0.992 on Overture, 1.004
        on OSM, 0.994 on matched, 0.996 pooled (equal segment weights);
      - arm C has the best pooled LPD per row (−0.441 vs −0.443);
      - arm B: 1.026 on matched.
17. **Paused for CPU** (Nat, 2026-09-28 11:28). No new model runs; running jobs
    finish; nothing killed.
    - SIGSTOP on the launchers only:
      - driver `run_bayes_phase1.sh`: PIDs 719105, 719181, 719182;
      - coverage orchestrator: PID 719180.
    - The heartbeat cron was deleted. It would otherwise try to restart a stalled
      pipeline.
    - At the pause: 14/50 coverage fits and 8/10 sensitivity fits were done. S5_gap01,
      S6_taumatched4 and four coverage workers were left to finish.
    - **Resume:** `kill -CONT 719180 719105 719181 719182`. The orchestrator then
      reaps its finished workers and launches the rest; the driver continues with S7
      and stage 4. Re-arm the heartbeat cron.
    - Mishap during the pause: an unbracketed `pgrep -f` matched my own shell and
      stopped it. It was resumed with SIGCONT, and no job was affected.
18. **Pipeline resumed and finished** (2026-09-29).
    - After the pause the machine restarted and the paused launchers were gone.
      The driver was relaunched and resumed cleanly: stages 1–2 skipped, 32 coverage
      fits and 3 sensitivity fits run. `PIPELINE DONE` at 23:29.
    - Report: `fit_report.md`. Section 4b (sensitivity shifts measured at the
      validation rows) was first appended by hand; on 2026-09-30 the report script
      was extended to generate it (decision 22), and the report was regenerated
      without re-running any fit.
19. **Headline results.**
    - **CV (10-fold, design-weighted; stratified bootstrap 95% intervals).** Arm C vs
      production as published:
      - relative Brier 0.992 [0.982, 1.003] Overture, 1.004 [0.998, 1.010] OSM,
        0.994 [0.978, 1.009] matched, **0.996 [0.990, 1.002] pooled**;
      - ΔLPD per row +0.0026 [−0.0006, +0.0058] pooled.
      - That is a statistical tie: slightly better in point terms on Overture and
        matched, a hair worse on OSM.
      - Arm C beats arm B: pooled 0.991 [0.986, 0.997], matched 0.969.
      - OOS RMSE pooled: C 0.3703, production 0.3709. OOS LPD summed mbg-style:
        C −1117.1, production −1123.5.
    - **Coverage of the arm C 95% band, in-family:** Overture 0.953, OSM 0.906,
      **matched 0.752** (flat region 0.665). Bias is negligible (|bias| ≤ 0.0014). The
      matched band is too narrow: the smoothing-prior under-coverage §6.11
      anticipated.
    - **Robustness.**
      - Under the "realistic" LLM (verdicts from arm A's fitted asymmetric,
        score-dependent θ), arm C is **biased upward**: matched +0.018, Overture
        +0.021, with matched coverage 0.27. Arm B stays near unbiased (matched
        −0.006, coverage 0.81).
      - Category confounder: arm C is unbiased, but matched coverage drops to 0.68.
      - Step at the atom: arm C matched 0.79, OSM 0.81.
      - Caveat: the realistic θ comes from arm A, whose own fit is biased. It may
        exaggerate the LLM's real error; in the gold data, exists verdicts are 98.5%
        correct and gone verdicts 99%.
    - **Sensitivity.** Every S3C–S7 run moves the curves by ≤ 0.006 on average at the
      validation rows (p95 ≤ 0.015). S2C does not (decision 15).
    - **Deployed impact vs October production** (20260902 population):
      - mean |Δ| 0.038 Overture, 0.010 OSM, 0.017 matched;
      - 12.5%, 0.9% and 5.5% of POIs move by more than 0.05;
      - Overture band crossings reach 49.9%, because the Overture floor sits near
        the 0.3 / 0.7 band edges.

20. **Answers to Q-a to Q-e and a correction** (Nat, 2026-09-30).
    - **Q-a:** the centre anchor with N(0, 1.5²) is kept.
    - **Q-b:** the sum-to-zero, centred measurement layer is accepted.
    - **Q-e:** a standard Horvitz–Thompson check will be designed in TODO.md and
      implemented in October.
    - **Q-c evidence.** Arm C's offset from the design-weighted gold rate matches the
      silver-label error measured on the gold concordance almost exactly:
      - Overture: arm C +0.013 vs HT; the exists verdicts wrong 5/181 imply −0.0147.
      - Matched: +0.009; wrong 3/231 implies −0.0102.
      - OSM: 0.000; the implied shift is −0.001 (exists wrong 1/191, gone wrong
        3/104).
      - So β ≈ 1 costs about +0.01 of upward bias where the LLM's exists verdicts
        err. That argues for pinning silver noise to the gold concordance by segment
        and verdict rather than fixing β = 1.
    - **Q-d diagnosis.** The matched bands are the right width for sampling noise:
      mean half-width 0.018 against 1.96 × the SD of the estimate, 0.0165. Coverage
      fails because of pointwise bias (corr(coverage, |bias| / half-width) = −0.96):
      - mean |bias| / half-width is 0.84 in the flat region and 0.54 elsewhere;
      - the worst points sit where the true surface approaches its ~0.94–0.96
        ceiling, with bias −0.03 to −0.09;
      - that is smoothing shrinkage at the sharp rise to the ceiling (the Cox
        phenomenon).
    - **Correction: "band change".** The deployed-impact column counts crossings of
      the 0.3/0.7/0.9 edges inherited from `compare_matched_index.BAND_EDGES`. The site
      has **no discrete bands**: `site/src/utils.js` maps confidence to a continuous
      Turbo gradient discretised to 0.05 steps.
      - The 49.9% "Overture band change" is almost entirely the 0.919912 atom
        (39% of unflagged Overture POIs): production 0.726, arm C 0.679, Δ −0.047.
        That straddles 0.7, and on the site it is a one-step colour change.
      - Plus the 0.990219 atom and 1.0 (~10%): production 0.827, arm C 0.923, Δ +0.096,
        about two steps.
21. **Arm C's silver-label accuracy becomes data** (Nat, 2026-09-30; design doc §3.5b,
    M15c'–d'). Wired in; not refitted.
    - `ModelSpec.label_noise = "fixed"` (the new default):
      - q_{g,ℓ} = P(exists | segment g, definitive LLM verdict ℓ), the design-weighted
        (Hájek, w = 1/π over production refined classes, per round) gold rate,
        Jeffreys-smoothed on its Kish ESS: (r·n_eff + 0.5) / (n_eff + 1);
      - each silver row enters as a fractional label, q·log p + (1 − q)·log(1 − p);
      - q is passed in as data (`prepare_data(..., silver_rates = ...)`, stored on
        `PreparedData.silver_rates`, recorded in each fit's `summary.json`).
    - **Direction.** It is the reverse, "P(true label | data label)", direction that
      Nat first specified. Gold measures it directly because phase 2 is uniform within
      verdict class. The forward (Se, Sp) direction would need an assumed prevalence to
      convert.
    - **Rates on round 20260730:**
      - Overture: exists 0.9732 (raw 0.9759, n 181), gone 0.0034 (0/148);
      - OSM: exists 0.9915 (raw 0.9941, n 191), gone 0.0346 (raw 0.0302, n 104);
      - matched: exists 0.9861 (raw 0.9882, n 231), gone 0.0086 (0/59).
    - **Folding in new rounds.** `silver_label_rates` pools any number of validation
      frames, each weighted by its own design. The scripts read
      `conflation.calibration.concordance_rounds` in config.yaml (empty now), overridable
      with `--concordance-rounds`. For October: `versions.calibration` becomes the
      October round and `concordance_rounds: ["20260730"]`. Extra rounds contribute all
      their gold; the current round contributes its training gold only. (Superseded by
      entry 23: the key is now `pooled_rounds`, and pooled rounds join the curve fit
      too.)
    - **CV.** The rates are recomputed from each fold's training gold (held-out labels
      excluded).
    - **Coverage study.** The in_family generator now uses the design-weighted forward
      rates (Se_g, Sp_g) when the truth fit has no β: Overture (1.000, 0.921), OSM
      (0.995, 0.965), matched (1.000, 0.861).
    - The estimated variants (`symmetric`, `asymmetric`, `none`) are kept as
      sensitivities. Every Phase 1 result is from the symmetric first run.
    - Tests 15–16 were added (rates by hand, holdout, pooling, fractional-label
      identity). 16 of 16 pass.
22. **Housekeeping at the end of Phase 1** (2026-09-30).
    - The design doc is rewritten to the final state: §0 and §3.5b; §3.7 as run; §4
      files; §5 as run; §7 as executed; §8 fold-in and drift check; §9 decisions 13–19;
      new §11 results and §12 open items.
    - The matched-band under-coverage (Q-d) is in TODO.md for before November (Nat's
      decision 19).
    - The HT check (Q-e) is designed in TODO.md for the October run.
    - An October-run checklist is added to TODO.md and linked from the run skills.
    - `report_bayes_calibration.py` now generates what was hand-appended: the at-row
      sensitivity table, the note on the 0.3 / 0.7 / 0.9 "band" column, and a
      silver-rates table for fixed-rate fits. `fit_report.md` was regenerated
      (report assembly only; the numbers are unchanged).
    - `run_bayes_phase1.sh` takes `EVAL` from the environment, so the October
      re-evaluation writes to a new directory. The report script's `gold_rate_table`
      and at-row table rebuild fixed-rate fits with the rates saved in their summary.

23. **Validation rounds pool into the curves, not only the rates** (Nat, 2026-10-01;
    design doc decision 20, §3.5b). Wired in; nothing re-run.
    - **Config.** `conflation.calibration.concordance_rounds` is renamed
      `pooled_rounds` (CLI `--pooled-rounds`; "" pools nothing), since it now governs
      the whole fit.
    - **Loading.** `bayes_calibration_common.load_handoff` returns the current round
      plus every pooled round, tagged by `validation_round`. `fit_rows` rebuilds an
      earlier fit's exact table from the `rounds` list its `summary.json` now records;
      for Phase 1 fits (no list) it loads the current round alone and checks the row
      count.
    - **Design per round.** `calibration_bayes.production_classes` builds the refined
      classes within each round and prefixes them with the round id. Every consumer
      groups by class, so inclusion probabilities, arm B's weights, the silver-label
      rates and the HT reference become per round with no other change. A single-round
      table keeps the unprefixed production classes, so all Phase 1 results reproduce.
      `silver_rates_for` was removed: `prepare_data` computes the rates from the pooled
      table's training gold.
    - **CV** (`current_round_folds`). Folds are assigned on the current round alone,
      exactly as production's cross-fit sees it. Pooled rounds get fold −1, so they
      stay in every training fit and are never scored. The comparator runs on the
      current round, and its row ids are mapped to the pooled table. On round 20260730
      alone the new fold array equals the old one, and the saved comparator folds
      agree.
    - **Coverage study.** Phase 2 is simulated per round at each round's realized
      inclusion; the simulator reads the rounds from the truth fit's summary.
    - **Fixed.** The CV worker read `logit_beta_label` for every arm C fit; a
      fixed-rate fit has none, so it would have crashed. It is now guarded.
    - **Test 17** checks that pooled classes are the per-round classes prefixed, that
      rates from the pooled table equal rates from the per-round frames, and that
      `prepare_data` uses them. 17 of 17 pass.
    - **Knots** come from the pooled phase-1 deciles (the decision-5 rule, unchanged).
    - **New drift check** (design doc §8): each round's design-weighted gold rate by
      score bin, before pooling. A systematic gap means the score's meaning moved (a
      matcher, CD or turnover-model change), and the rounds should not be pooled.
24. **Answers to the 2026-09-30 questions** (Nat, 2026-10-01).
    - **Rate cells:** stay segment × verdict (design doc decision 21).
    - **Curves pool rounds:** yes (entry 23).
    - **Fractional labels vs a mixture, and score-dependent q:** Nat asked for more
      detail. Written up in design doc §3.5c with a tercile check on round 20260730:
      no tercile differs significantly from its cell's q (largest high − low |z| 1.4,
      on Overture "exists"). Decision pending.
    - **HT check:** never fails a run; a review document with graphics (PDF), marking
      bins more than ±1 SD from the bin's exists/checked rate (design doc decision 22;
      TODO.md).
    - **CHANGELOG:** first no entry while the prototype does not change published
      data; later the same day Nat asked for one. An "evaluated, not deployed" entry
      now sits under Unreleased (design doc decision 23).
    - **Commit:** held.

25. **Answers on the mixture, q and the HT reference** (Nat, 2026-10-01).
    - **Fixed-rate mixture:** added to the October TODOs as a test model beside the
      fractional-label arm C (design doc decision 24). Planned as `label_noise =
      "fixed_mixture"`, with design-weighted, Jeffreys-smoothed forward rates from the
      same training gold as q; not built yet.
    - **q by score:** not varied (design doc decision 25).
    - **HT flag reference:** the design-weighted ratio with SD from the Kish ESS, the
      raw ratio alongside, and the chance baselines (design doc decision 26).
    - **Library:** Nat added Louis (1982) [U2LXHF6H] and Orchard & Woodbury (1972)
      [KDGF5K5J]; the design doc's references carry the IDs.
26. **The fixed-rate mixture is built** (2026-10-01; design doc §3.5c, decision 24).
    Code and tests only; nothing fitted.
    - **Likelihood.** `ModelSpec.label_noise = "fixed_mixture"` takes the (M15c)
      branch of `pointwise_log_likelihood`: log[p Se + (1 − p)(1 − Sp)] for
      "exists", log[p(1 − Se) + (1 − p) Sp] for "gone", and the plain Bernoulli for
      gold. Se and Sp are per-segment data (`silver_se`, `silver_sp` in the
      prepared segments), so no parameter or prior is added. A rate of exactly 0 or
      1 is floored at log −1e3, as for "none".
    - **Rates.** `forward_silver_rates` moved from `simulate_bayes_recovery.py` into
      `calibration_bayes` beside `silver_label_rates`. Both now share one weighting
      helper, so the forward rates use the same per-round 1/π weights, round pooling
      and training-gold mask as q. Each rate is Jeffreys-smoothed on the Kish ESS of
      its denominator (the gold rows that exist, for Se; those that do not, for Sp).
      `prepare_data` computes them from the fit's training gold unless
      `forward_rates` is passed. The fit stores them on
      `PreparedData.forward_rates` and in `summary.json`, and the report rebuilds
      fits from the saved rates.
    - **Simulator.** It imports the library function. The in_family generator keeps
      the raw rates it used in Phase 1 (Overture Se = 1.000); a fitted mixture gets
      the smoothed ones. `--label-noise` now reaches the fitted arm C, and
      `--tag-suffix` names a variant's files `<scenario>_rNN_armC<suffix>` and
      labels its summary rows `C<suffix>`. The truth stays the `armC` fit.
    - **CV.** New `--compare-tags` assembles earlier passes' folds (e.g.
      `armC,armB`) beside a suffixed pass. The paired bootstrap then compares the
      mixture with arm C and arm B, and writes `cv_results_mixture.*`.
    - **Tests 18–20.** Se = Sp = 1 gives the Bernoulli on the label. Value and
      gradient match a numpy reference: the analytic dF = r − p pulled back through
      the curves, and finite differences. The forward rates match a hand
      computation, ignore held-out gold, pool rounds, and stay strictly inside
      (0, 1) when the raw rates are 1.
## Questions for Nat: status

- **Q-a** (data-centre anchor, α ~ N(0, 1.5²)): accepted, 2026-09-30.
- **Q-b** (centred, sum-to-zero measurement layer): accepted, 2026-09-30.
- **Q-c** (arm C's noise, β = 1 vs gold concordance): resolved to fixed gold-concordance
  rates as data (decision 21).
- **Q-d** (matched under-coverage): diagnosed as smoothing bias (decision 20). TODO
  before November.
- **Q-e** (HT guard): adopted as a standard run-cycle output; designed in TODO.md for
  October.
- **2026-09-30 round** (rate cells, pooled curves, HT format, CHANGELOG, commit):
  answered 2026-10-01 (entry 24).
- **Q-f** (fractional labels vs the fixed-rate mixture; score-dependent q): resolved
  2026-10-01 (entry 25).
- **Q-g** (HT flag reference): resolved 2026-10-01 (entry 25).
- Still open: see the design doc §12.
