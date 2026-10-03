#!/bin/bash
# Congestion sweep (2x) on intas_crop (Ingolstadt real-network crop).
# 1x baseline already exists in output/intas_crop_bo_vanilla_matern-2p5/.
# Seeds: 42, 1042, 2042 (restart 1, 2, 3).
# Logs → scripts/logs/
#
# NOTE: setup_congestion_level_intas.py --demand_scale 2.0 has already been run
# (output/intas_crop_congestion_2x/ exists with GT sim + 20 init points). The
# setup step below is safe to re-run (idempotent, just regenerates the same
# GT sim + init points) but you can comment it out to skip straight to BO.
set -e
cd /Users/inesbenhamza/Desktop/Sumo_od_calibration_bn
mkdir -p scripts/logs

# ── Setup (sequential: GT sim must finish before BO starts) ───────────────────
echo "=== SETUP 2x ==="
python3 scripts/setup/setup_congestion_level_intas.py --demand_scale 2.0 \
    > scripts/logs/setup_intas_2x.log 2>&1

echo "=== Setup done — launching BO runs ==="

run_batch() {
    echo "=== starting batch: $* ==="
    for cmd in "$@"; do
        eval "$cmd" &
    done
    wait
    echo "=== batch done ==="
}

# Batch 1: IndGP 2x (3 seeds)
run_batch \
    "python3 scripts/models/indgp/run_congestion_indgp_bo_intas.py --demand_scale 2.0 --seed 42   > scripts/logs/indgp_intas_2x_s42.log   2>&1" \
    "python3 scripts/models/indgp/run_congestion_indgp_bo_intas.py --demand_scale 2.0 --seed 1042 > scripts/logs/indgp_intas_2x_s1042.log 2>&1" \
    "python3 scripts/models/indgp/run_congestion_indgp_bo_intas.py --demand_scale 2.0 --seed 2042 > scripts/logs/indgp_intas_2x_s2042.log 2>&1"

# Batch 2: ScalarGP 2x (3 seeds)
run_batch \
    "python3 scripts/models/scalargp/run_congestion_scalargp_bo_intas.py --demand_scale 2.0 --seed 42   > scripts/logs/scalargp_intas_2x_s42.log   2>&1" \
    "python3 scripts/models/scalargp/run_congestion_scalargp_bo_intas.py --demand_scale 2.0 --seed 1042 > scripts/logs/scalargp_intas_2x_s1042.log 2>&1" \
    "python3 scripts/models/scalargp/run_congestion_scalargp_bo_intas.py --demand_scale 2.0 --seed 2042 > scripts/logs/scalargp_intas_2x_s2042.log 2>&1"

# Batch 3: GraphGP-Matern 2x (3 seeds)
run_batch \
    "python3 scripts/models/route_overlap_graphgp/run_congestion_graphgp_matern_bo_intas.py --demand_scale 2.0 --seed 42   > scripts/logs/graphgp_intas_2x_s42.log   2>&1" \
    "python3 scripts/models/route_overlap_graphgp/run_congestion_graphgp_matern_bo_intas.py --demand_scale 2.0 --seed 1042 > scripts/logs/graphgp_intas_2x_s1042.log 2>&1" \
    "python3 scripts/models/route_overlap_graphgp/run_congestion_graphgp_matern_bo_intas.py --demand_scale 2.0 --seed 2042 > scripts/logs/graphgp_intas_2x_s2042.log 2>&1"

# Batch 4: ICM (rank=1) 2x (3 seeds)
run_batch \
    "python3 scripts/models/icm_kronecker/run_congestion_icm_bo_intas.py --demand_scale 2.0 --seed 42   > scripts/logs/icm_intas_2x_s42.log   2>&1" \
    "python3 scripts/models/icm_kronecker/run_congestion_icm_bo_intas.py --demand_scale 2.0 --seed 1042 > scripts/logs/icm_intas_2x_s1042.log 2>&1" \
    "python3 scripts/models/icm_kronecker/run_congestion_icm_bo_intas.py --demand_scale 2.0 --seed 2042 > scripts/logs/icm_intas_2x_s2042.log 2>&1"

# Batch 5: Shared+Private 2x (3 seeds)
run_batch \
    "python3 scripts/models/shared_private/run_congestion_shared_private_bo_intas.py --demand_scale 2.0 --seed 42   > scripts/logs/sharedprivate_intas_2x_s42.log   2>&1" \
    "python3 scripts/models/shared_private/run_congestion_shared_private_bo_intas.py --demand_scale 2.0 --seed 1042 > scripts/logs/sharedprivate_intas_2x_s1042.log 2>&1" \
    "python3 scripts/models/shared_private/run_congestion_shared_private_bo_intas.py --demand_scale 2.0 --seed 2042 > scripts/logs/sharedprivate_intas_2x_s2042.log 2>&1"

echo "ALL INTAS_CROP CONGESTION RUNS COMPLETE"
