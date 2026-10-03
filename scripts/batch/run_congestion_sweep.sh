#!/bin/bash
# Congestion sweep: grid_2d_3x3 at 0.5x and 2x demand.
# Compares IndGP vs GraphGP-Matérn — the 1x baseline already exists.
#
# Step 1: setup GT + init points for each new congestion level (sequential)
# Step 2: run BO for both methods × 2 scales × 3 seeds (parallel batches)
set -e
cd /Users/inesbenhamza/Desktop/Sumo_od_calibration_bn
mkdir -p scripts/logs

echo "=== SETUP: demand 0.5x ==="
python3 scripts/setup_congestion_level.py --demand_scale 0.5 \
  > scripts/logs/setup_congestion_0.5x.log 2>&1
echo "    done. $(tail -1 scripts/logs/setup_congestion_0.5x.log)"

echo "=== SETUP: demand 2x ==="
python3 scripts/setup_congestion_level.py --demand_scale 2.0 \
  > scripts/logs/setup_congestion_2x.log 2>&1
echo "    done. $(tail -1 scripts/logs/setup_congestion_2x.log)"

run_batch() {
    echo "=== batch: $* ==="
    for cmd in "$@"; do eval "$cmd" & done
    wait
    echo "=== batch done ==="
}

# Batch 1: 0.5x seeds 42/1042/2042 — both methods (6 parallel)
run_batch \
  "python3 scripts/models/indgp/run_congestion_indgp_bo.py      --demand_scale 0.5 --seed 42   --max_iters 50 > scripts/logs/congestion_indgp_0.5x_seed42.log   2>&1" \
  "python3 scripts/models/indgp/run_congestion_indgp_bo.py      --demand_scale 0.5 --seed 1042 --max_iters 50 > scripts/logs/congestion_indgp_0.5x_seed1042.log 2>&1" \
  "python3 scripts/models/indgp/run_congestion_indgp_bo.py      --demand_scale 0.5 --seed 2042 --max_iters 50 > scripts/logs/congestion_indgp_0.5x_seed2042.log 2>&1" \
  "python3 scripts/models/route_overlap_graphgp/run_congestion_graphgp_matern_bo.py --demand_scale 0.5 --seed 42   --max_iters 50 > scripts/logs/congestion_graphgp_0.5x_seed42.log   2>&1" \
  "python3 scripts/models/route_overlap_graphgp/run_congestion_graphgp_matern_bo.py --demand_scale 0.5 --seed 1042 --max_iters 50 > scripts/logs/congestion_graphgp_0.5x_seed1042.log 2>&1" \
  "python3 scripts/models/route_overlap_graphgp/run_congestion_graphgp_matern_bo.py --demand_scale 0.5 --seed 2042 --max_iters 50 > scripts/logs/congestion_graphgp_0.5x_seed2042.log 2>&1"

# Batch 2: 2x seeds 42/1042/2042 — both methods (6 parallel)
run_batch \
  "python3 scripts/models/indgp/run_congestion_indgp_bo.py      --demand_scale 2.0 --seed 42   --max_iters 50 > scripts/logs/congestion_indgp_2x_seed42.log   2>&1" \
  "python3 scripts/models/indgp/run_congestion_indgp_bo.py      --demand_scale 2.0 --seed 1042 --max_iters 50 > scripts/logs/congestion_indgp_2x_seed1042.log 2>&1" \
  "python3 scripts/models/indgp/run_congestion_indgp_bo.py      --demand_scale 2.0 --seed 2042 --max_iters 50 > scripts/logs/congestion_indgp_2x_seed2042.log 2>&1" \
  "python3 scripts/models/route_overlap_graphgp/run_congestion_graphgp_matern_bo.py --demand_scale 2.0 --seed 42   --max_iters 50 > scripts/logs/congestion_graphgp_2x_seed42.log   2>&1" \
  "python3 scripts/models/route_overlap_graphgp/run_congestion_graphgp_matern_bo.py --demand_scale 2.0 --seed 1042 --max_iters 50 > scripts/logs/congestion_graphgp_2x_seed1042.log 2>&1" \
  "python3 scripts/models/route_overlap_graphgp/run_congestion_graphgp_matern_bo.py --demand_scale 2.0 --seed 2042 --max_iters 50 > scripts/logs/congestion_graphgp_2x_seed2042.log 2>&1"

echo "ALL CONGESTION SWEEP RUNS COMPLETE"
