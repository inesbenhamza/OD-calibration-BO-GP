#!/bin/bash
# Runs plain ICM (rank=1, N_INIT=20) seeds 1042-4042 and Graph-ICM (rank=4,
# N_INIT=20, route-overlap warm-start) seeds 42-4042, in batches of 3
# parallel processes. Seed 42 of plain ICM already done.
set -e
cd /Users/inesbenhamza/Desktop/Sumo_od_calibration_bn
mkdir -p scripts/logs

run_batch() {
    echo "=== starting batch: $* ==="
    for cmd in "$@"; do
        eval "$cmd" &
    done
    wait
    echo "=== batch done ==="
}

# Batch 1: ICM rank=1, seeds 1042-3042
run_batch \
    "python3 scripts/models/icm_kronecker/run_icm_mc_ei_bo.py --seed 1042 --rank 1 --n_init 20 --max_iters 50 > scripts/logs/icm_r1_seed1042.log 2>&1" \
    "python3 scripts/models/icm_kronecker/run_icm_mc_ei_bo.py --seed 2042 --rank 1 --n_init 20 --max_iters 50 > scripts/logs/icm_r1_seed2042.log 2>&1" \
    "python3 scripts/models/icm_kronecker/run_icm_mc_ei_bo.py --seed 3042 --rank 1 --n_init 20 --max_iters 50 > scripts/logs/icm_r1_seed3042.log 2>&1"

# Batch 2: ICM rank=1 seed 4042 + Graph-ICM rank=4, seeds 42-1042
run_batch \
    "python3 scripts/models/icm_kronecker/run_icm_mc_ei_bo.py --seed 4042 --rank 1 --n_init 20 --max_iters 50 > scripts/logs/icm_r1_seed4042.log 2>&1" \
    "python3 scripts/models/icm_kronecker/run_graphicm_bo_grid3.py --seed 42  --rank 4 --n_init 20 --max_iters 50 > scripts/logs/graphicm_r4_seed42.log 2>&1" \
    "python3 scripts/models/icm_kronecker/run_graphicm_bo_grid3.py --seed 1042 --rank 4 --n_init 20 --max_iters 50 > scripts/logs/graphicm_r4_seed1042.log 2>&1"

# Batch 3: Graph-ICM rank=4, seeds 2042-4042
run_batch \
    "python3 scripts/models/icm_kronecker/run_graphicm_bo_grid3.py --seed 2042 --rank 4 --n_init 20 --max_iters 50 > scripts/logs/graphicm_r4_seed2042.log 2>&1" \
    "python3 scripts/models/icm_kronecker/run_graphicm_bo_grid3.py --seed 3042 --rank 4 --n_init 20 --max_iters 50 > scripts/logs/graphicm_r4_seed3042.log 2>&1" \
    "python3 scripts/models/icm_kronecker/run_graphicm_bo_grid3.py --seed 4042 --rank 4 --n_init 20 --max_iters 50 > scripts/logs/graphicm_r4_seed4042.log 2>&1"

echo "ALL ICM / GRAPH-ICM COMPARISON RUNS COMPLETE"
