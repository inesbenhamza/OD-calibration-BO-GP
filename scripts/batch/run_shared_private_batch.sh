#!/bin/bash
# Shared+Private GraphGP batch runs (quickstart_grid_3 only -- the grid_2d_3x3
# simplified-kernel script this used to also run was removed: it used a
# different, non-per-edge-lengthscale kernel than MOGP.graph.SharedPrivateGraphKernel,
# see scripts/models/shared_private/run_congestion_sharedprivate_signedresidual_bo.py
# for the real-kernel grid_2d_3x3 congestion script instead)
# Runs 2x first (priority), then 0.5x
# 3 seeds per scale: 42, 1042, 2042

set -e
PYTHON="python3"
SP_Q="scripts/models/shared_private/run_congestion_shared_private_bo_qg3.py"
LOG_DIR="scripts/logs"
mkdir -p "$LOG_DIR"

echo "====  Batch 1: 2× — quickstart_grid_3 (3 seeds)  ===="
$PYTHON $SP_Q  --demand_scale 2.0 --seed 42   > "$LOG_DIR/sp_q_2x_s42.log"   2>&1 &
$PYTHON $SP_Q  --demand_scale 2.0 --seed 1042 > "$LOG_DIR/sp_q_2x_s1042.log" 2>&1 &
$PYTHON $SP_Q  --demand_scale 2.0 --seed 2042 > "$LOG_DIR/sp_q_2x_s2042.log" 2>&1 &
wait
echo "Batch 1 done."

echo "====  Batch 2: 0.5× — quickstart_grid_3 (3 seeds)  ===="
$PYTHON $SP_Q  --demand_scale 0.5 --seed 42   > "$LOG_DIR/sp_q_0p5x_s42.log"   2>&1 &
$PYTHON $SP_Q  --demand_scale 0.5 --seed 1042 > "$LOG_DIR/sp_q_0p5x_s1042.log" 2>&1 &
$PYTHON $SP_Q  --demand_scale 0.5 --seed 2042 > "$LOG_DIR/sp_q_0p5x_s2042.log" 2>&1 &
wait
echo "Batch 2 done."

echo "==== ALL DONE ===="
