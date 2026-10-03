#!/bin/bash
# Runs the remaining quickstart_grid_3 comparison jobs (3 seeds each for
# diffusion/regularized/matern graph kernels + IndGP), in batches of 3
# parallel processes at a time to keep CPU contention bounded on an 8-core
# machine. Seeds follow this project's existing notebook convention
# (restart_seed = 42 + r*1000 for r in 0,1,2 -> 42, 1042, 2042). Logs go to
# scripts/logs/.
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

# Batch 1
run_batch \
    "python3 scripts/models/route_overlap_graphgp/run_signed_residual_mc_ei_bo.py --kernel_type diffusion --seed 1042 > scripts/logs/diffusion_seed1042.log 2>&1" \
    "python3 scripts/models/route_overlap_graphgp/run_signed_residual_mc_ei_bo.py --kernel_type diffusion --seed 2042 > scripts/logs/diffusion_seed2042.log 2>&1" \
    "python3 scripts/models/route_overlap_graphgp/run_signed_residual_mc_ei_bo.py --kernel_type regularized_laplacian --seed 42 > scripts/logs/regularized_seed42.log 2>&1"

# Batch 2
run_batch \
    "python3 scripts/models/route_overlap_graphgp/run_signed_residual_mc_ei_bo.py --kernel_type regularized_laplacian --seed 1042 > scripts/logs/regularized_seed1042.log 2>&1" \
    "python3 scripts/models/route_overlap_graphgp/run_signed_residual_mc_ei_bo.py --kernel_type regularized_laplacian --seed 2042 > scripts/logs/regularized_seed2042.log 2>&1" \
    "python3 scripts/models/route_overlap_graphgp/run_signed_residual_mc_ei_bo.py --kernel_type matern --seed 42 > scripts/logs/matern_seed42.log 2>&1"

# Batch 3
run_batch \
    "python3 scripts/models/route_overlap_graphgp/run_signed_residual_mc_ei_bo.py --kernel_type matern --seed 1042 > scripts/logs/matern_seed1042.log 2>&1" \
    "python3 scripts/models/route_overlap_graphgp/run_signed_residual_mc_ei_bo.py --kernel_type matern --seed 2042 > scripts/logs/matern_seed2042.log 2>&1" \
    "python3 scripts/models/indgp/run_indgp_mc_ei_bo.py --seed 42 > scripts/logs/indgp_seed42.log 2>&1"

# Batch 4
run_batch \
    "python3 scripts/models/indgp/run_indgp_mc_ei_bo.py --seed 1042 > scripts/logs/indgp_seed1042.log 2>&1" \
    "python3 scripts/models/indgp/run_indgp_mc_ei_bo.py --seed 2042 > scripts/logs/indgp_seed2042.log 2>&1"

echo "ALL QUICKSTART_GRID_3 BATCH RUNS COMPLETE"
