#!/bin/bash
# Congestion sweep (0.5x, 2x) on quickstart_grid_3.
# 1x baseline already exists in output/quickstart_grid_3_bo_vanilla_matern-2p5/.
# Seeds: 42, 1042, 2042 (restart 1, 2, 3).
# Logs → scripts/logs/
set -e
cd /Users/inesbenhamza/Desktop/Sumo_od_calibration_bn
mkdir -p scripts/logs

# ── Setup (sequential: GT sim must finish before BO starts) ───────────────────
echo "=== SETUP 0.5x ==="
python3 scripts/setup_congestion_level_qg3.py --demand_scale 0.5 \
    > scripts/logs/setup_qg3_0.5x.log 2>&1
echo "=== SETUP 2x ==="
python3 scripts/setup_congestion_level_qg3.py --demand_scale 2.0 \
    > scripts/logs/setup_qg3_2x.log 2>&1

echo "=== Setups done — launching BO runs ==="

run_batch() {
    echo "=== starting batch: $* ==="
    for cmd in "$@"; do
        eval "$cmd" &
    done
    wait
    echo "=== batch done ==="
}

# Batch 1: IndGP 0.5x (3 seeds)
run_batch \
    "python3 scripts/models/indgp/run_congestion_indgp_bo_qg3.py --demand_scale 0.5 --seed 42   > scripts/logs/indgp_qg3_0.5x_s42.log 2>&1" \
    "python3 scripts/models/indgp/run_congestion_indgp_bo_qg3.py --demand_scale 0.5 --seed 1042 > scripts/logs/indgp_qg3_0.5x_s1042.log 2>&1" \
    "python3 scripts/models/indgp/run_congestion_indgp_bo_qg3.py --demand_scale 0.5 --seed 2042 > scripts/logs/indgp_qg3_0.5x_s2042.log 2>&1"

# Batch 2: IndGP 2x (3 seeds)
run_batch \
    "python3 scripts/models/indgp/run_congestion_indgp_bo_qg3.py --demand_scale 2.0 --seed 42   > scripts/logs/indgp_qg3_2x_s42.log 2>&1" \
    "python3 scripts/models/indgp/run_congestion_indgp_bo_qg3.py --demand_scale 2.0 --seed 1042 > scripts/logs/indgp_qg3_2x_s1042.log 2>&1" \
    "python3 scripts/models/indgp/run_congestion_indgp_bo_qg3.py --demand_scale 2.0 --seed 2042 > scripts/logs/indgp_qg3_2x_s2042.log 2>&1"

# Batch 3: GraphGP 0.5x (3 seeds)
run_batch \
    "python3 scripts/models/route_overlap_graphgp/run_congestion_graphgp_matern_bo_qg3.py --demand_scale 0.5 --seed 42   > scripts/logs/graphgp_qg3_0.5x_s42.log 2>&1" \
    "python3 scripts/models/route_overlap_graphgp/run_congestion_graphgp_matern_bo_qg3.py --demand_scale 0.5 --seed 1042 > scripts/logs/graphgp_qg3_0.5x_s1042.log 2>&1" \
    "python3 scripts/models/route_overlap_graphgp/run_congestion_graphgp_matern_bo_qg3.py --demand_scale 0.5 --seed 2042 > scripts/logs/graphgp_qg3_0.5x_s2042.log 2>&1"

# Batch 4: GraphGP 2x (3 seeds)
run_batch \
    "python3 scripts/models/route_overlap_graphgp/run_congestion_graphgp_matern_bo_qg3.py --demand_scale 2.0 --seed 42   > scripts/logs/graphgp_qg3_2x_s42.log 2>&1" \
    "python3 scripts/models/route_overlap_graphgp/run_congestion_graphgp_matern_bo_qg3.py --demand_scale 2.0 --seed 1042 > scripts/logs/graphgp_qg3_2x_s1042.log 2>&1" \
    "python3 scripts/models/route_overlap_graphgp/run_congestion_graphgp_matern_bo_qg3.py --demand_scale 2.0 --seed 2042 > scripts/logs/graphgp_qg3_2x_s2042.log 2>&1"

echo "ALL QG3 CONGESTION RUNS COMPLETE"
