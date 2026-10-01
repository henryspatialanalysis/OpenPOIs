#!/usr/bin/env bash
# Unattended Phase 1 pipeline for the Bayesian calibration prototype, with arm C
# (the simple silver layer) as the main model (execution log, decision 12). Since
# decision 21 arm C's default is fixed silver-label rates (label_noise = fixed);
# the Phase 1 outputs in calibration_eval_bayes_20260927 were produced with the
# estimated symmetric beta, so a re-run should use a new EVAL directory.
#
# Stage 1: main full-data fits. Arm C (main) and arms A and B (comparators) are
#          fit once; arm C's PPC and deployed impact come from its saved draws.
#          The best arm A variant (ARM_A_FLAGS) gets its own full-data fit.
# Stage 2: 10-fold CV of arms C, B and the arm A variant against the production
#          cross-fit.
# Stage 3: fake-data coverage study (arm C fitted; arm B on misspecified rounds)
#          and the arm C sensitivity fits S2C-S7.
# Stage 3b: the fixed-rate mixture test model (label_noise = fixed_mixture, design
#          doc §3.5c): its full fit, a CV pass compared pairwise with arms C and B,
#          and its in-family coverage.
# Stage 4: fit_report.md.
#
# MODE=mixture is the production calibration from October 2026 on (Nat,
# 2026-09-30): three fixed-rate mixture fits, one per segment (tags
# mixture_overture, mixture_osm, mixture_matched), run in parallel with
# OMP_NUM_THREADS=2 each, then the report over all three. Under arm C with fixed
# rates the joint posterior factorizes by segment, so the three fits are the
# joint fit, each with its own step size and mass matrix. Any failed fit stops
# the run. Its EVAL defaults to <conflation root>/<versions.conflation>/
# calibration_bayes; export_bayes_curves.py turns the fits into deployable grid
# curves. MODE=full (default) runs every stage (joint fits, evaluation only).
#
# Every stage is resumable: the Python scripts skip outputs that already exist
# (CV folds, coverage rounds); full fits are skipped when summary.json exists,
# and a fit already running under the same tag is waited for, not duplicated.
# Log lines "STAGE <n> START/DONE" and "PIPELINE DONE/FAILED" are the markers the
# heartbeat watches.
set -u
REPO=~/repos/openpois
PY=~/miniforge3/envs/openpois/bin/python
MODE=${MODE:-full}
# Output directory: override for a new run, e.g.
#   EVAL=~/data/openpois/conflation/<round version>/calibration_eval_bayes_<date> bash run_bayes_phase1.sh
# MODE=mixture defaults to the run's own conflation version (read from config.yaml
# as the remote stage runner does).
if [ -z "${EVAL:-}" ]; then
  if [ "$MODE" = mixture ]; then
    VERSION=$(cd "$REPO" && $PY -c "import yaml; print(yaml.safe_load(open('config.yaml'))['versions']['conflation'])") \
      || { echo "PIPELINE FAILED: cannot read versions.conflation"; exit 1; }
    EVAL=~/data/openpois/conflation/$VERSION/calibration_bayes
  else
    EVAL=~/data/openpois/conflation/20260730/calibration_eval_bayes_20260927
  fi
fi
mkdir -p "$EVAL/logs"
LOGS=$EVAL/logs
# Main fits get 1,000 + 1,000; CV / coverage / sensitivity fits need posterior
# means and 95% bands only (execution log, decision 8). MAIN may be overridden
# for a short local check, e.g. MAIN="--warmup 200 --samples 200 --chains 2".
MAIN=${MAIN:-"--warmup 1000 --samples 1000 --chains 4"}
LIGHT="--warmup 600 --samples 400 --chains 4"
PARALLEL=${PARALLEL:-5}
# Flags of the arm A variant carried into CV (set from the structure tests).
ARM_A_FLAGS=${ARM_A_FLAGS:-}
ARM_A_TAG=${ARM_A_TAG:-armA_variant}
# CV arms. The arm A CV was deferred (Nat, 2026-09-28; execution log,
# decision 16): the default is C,B.
CV_ARMS=${CV_ARMS:-C,B}
COV_PARALLEL=${COV_PARALLEL:-5}
MIXTURE_TAG=armC_mixture
# MODE=mixture: one fit per segment, tag mixture_<segment>.
SEGMENTS="overture osm matched"
export OMP_NUM_THREADS=3
cd "$REPO"

stamp() { echo "[$(date '+%F %T')] $*"; }

fit() {  # fit <tag> <args...>
  local tag=$1; shift
  while pgrep -f "[f]it_bayes_calibration.py.*--tag $tag " > /dev/null; do
    sleep 60
  done
  if [ -f "$EVAL/fits/$tag/summary.json" ]; then
    stamp "fit $tag exists, skipping"; return 0
  fi
  $PY -u scripts/conflation/fit_bayes_calibration.py --out-dir "$EVAL" --tag "$tag" "$@" \
    > "$LOGS/fit_$tag.log" 2>&1
  local rc=$?
  stamp "fit $tag exit $rc"
  return $rc
}

if [ "$MODE" = mixture ]; then
  stamp "MODE mixture: one fixed-rate mixture fit per segment, then the report ($EVAL)"
  declare -A PIDS
  for s in $SEGMENTS; do
    ( export OMP_NUM_THREADS=2
      fit "mixture_$s" --arm C --label-noise fixed_mixture --segments "$s" $MAIN \
        --deployed-impact ) &
    PIDS[$s]=$!
  done
  FAILED=""
  for s in $SEGMENTS; do
    wait "${PIDS[$s]}" || FAILED="$FAILED mixture_$s"
    [ -f "$EVAL/fits/mixture_$s/summary.json" ] || FAILED="$FAILED mixture_$s"
  done
  if [ -n "$FAILED" ]; then
    stamp "PIPELINE FAILED:$FAILED"; exit 1
  fi
  $PY -u scripts/conflation/report_bayes_calibration.py --out-dir "$EVAL" \
    --main mixture_matched > "$LOGS/report.log" 2>&1 \
    || { stamp "PIPELINE FAILED: report"; exit 1; }
  stamp "PIPELINE DONE"
  exit 0
elif [ "$MODE" != full ]; then
  stamp "PIPELINE FAILED: unknown MODE $MODE (full or mixture)"; exit 1
fi

stamp "STAGE 1 START (main fits)"
# In the Phase 1 directory these fits exist already, so the calls only wait / skip;
# in a fresh EVAL they run.
fit armC --arm C $MAIN --deployed-impact
fit armA --arm A $MAIN
fit armB --arm B $MAIN
if [ -n "$ARM_A_FLAGS" ]; then
  fit "$ARM_A_TAG" --arm A $MAIN $ARM_A_FLAGS
fi
for t in armC armA armB; do
  [ -f "$EVAL/fits/$t/summary.json" ] || { stamp "PIPELINE FAILED: $t"; exit 1; }
done
stamp "STAGE 1 DONE"

stamp "STAGE 2 START (CV)"
$PY -u scripts/conflation/cv_bayes_calibration.py --out-dir "$EVAL" --arms "$CV_ARMS" \
  --arm-flags "A=$ARM_A_FLAGS" --parallel "$PARALLEL" $LIGHT \
  >> "$LOGS/cv.log" 2>&1 || { stamp "PIPELINE FAILED: CV"; exit 1; }
stamp "STAGE 2 DONE"

stamp "STAGE 3 START (coverage + sensitivity)"
$PY -u scripts/conflation/simulate_bayes_recovery.py --out-dir "$EVAL" --rounds 20 \
  --misspecified-rounds 5 --parallel "$COV_PARALLEL" $LIGHT > "$LOGS/coverage.log" 2>&1 &
COV=$!
run_sens() {
  fit S2C_asym --arm C $LIGHT --label-noise asymmetric
  fit S3C_exact --arm C $LIGHT --label-noise none
  # The beta prior only acts under estimated symmetric noise (arm C's default is
  # fixed rates since decision 21), so S3C-b states the variant explicitly.
  fit S3C_weak --arm C $LIGHT --label-noise symmetric --beta-prior 9,1
  fit S4_cubic --arm C $LIGHT --degree 3
  fit S5_gap01 --arm C $LIGHT --max-gap 0.1
}
run_sens2() {
  fit S5_equal20 --arm C $LIGHT --equal-knots 20
  fit S6_tau2 --arm C $LIGHT --tau-mult 2
  fit S6_tauhalf --arm C $LIGHT --tau-mult 0.5
  fit S6_taumatched4 --arm C $LIGHT --tau-matched-mult 4
  fit S7_spacing --arm C $LIGHT --spacing-scaled-rw
}
run_sens & S1=$!
run_sens2 & S2=$!
wait $S1 $S2
wait $COV || stamp "coverage exited non-zero (see coverage.log)"
stamp "STAGE 3 DONE"

stamp "STAGE 3b START (fixed-rate mixture)"
fit "$MIXTURE_TAG" --arm C --label-noise fixed_mixture $MAIN --deployed-impact \
  || { stamp "PIPELINE FAILED: $MIXTURE_TAG"; exit 1; }
# The CV pass reuses the armC and armB folds from stage 2 for the paired comparison.
$PY -u scripts/conflation/cv_bayes_calibration.py --out-dir "$EVAL" --arms C \
  --tag-suffix _mixture --label-noise fixed_mixture --compare-tags armC,armB \
  --parallel "$PARALLEL" $LIGHT >> "$LOGS/cv_mixture.log" 2>&1 \
  || { stamp "PIPELINE FAILED: mixture CV"; exit 1; }
$PY -u scripts/conflation/simulate_bayes_recovery.py --out-dir "$EVAL" --rounds 20 \
  --misspecified-rounds 0 --label-noise fixed_mixture --tag-suffix _mixture \
  --parallel "$COV_PARALLEL" $LIGHT > "$LOGS/coverage_mixture.log" 2>&1 \
  || stamp "mixture coverage exited non-zero (see coverage_mixture.log)"
stamp "STAGE 3b DONE"

stamp "STAGE 4 START (report)"
$PY -u scripts/conflation/report_bayes_calibration.py --out-dir "$EVAL" > "$LOGS/report.log" 2>&1 \
  || { stamp "PIPELINE FAILED: report"; exit 1; }
stamp "PIPELINE DONE"
