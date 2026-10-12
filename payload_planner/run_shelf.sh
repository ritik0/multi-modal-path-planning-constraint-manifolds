#!/usr/bin/env bash
# ===========================================================================
# run_shelf.sh -- two-level shelf scenario for the payload transport planner.
#
# This reuses run_panda.py unchanged; it only sets the PAYLOAD_SURF_* env vars
# that move the two support surfaces to different heights.  The object is slid
# off the LOWER shelf (surface A, z=0), lifted through free flight, and set down
# on the UPPER shelf (surface B, raised).  The wall obstacle is disabled here;
# the height difference between the two shelves is itself the reason free flight
# is needed, so no extra obstacle is required.  Enable the transfer foliation to
# let the planner choose the carry height between the two shelf levels.
#
# Geometry (kept inside the reach envelope the two-table runs already proved):
#   lower shelf A : z = 0.00,  x 0.09..0.38,  y  0.20..0.58   (right, low)
#   upper shelf B : z = 0.30,  x 0.09..0.38,  y -0.58..-0.20  (left,  high)
#
# Tune PAYLOAD_SURF_B_Z to make the shelf taller/shorter; raising it makes the
# lift harder (more torque to reach up) and is the knob for the shelf-height
# experiment.
# ===========================================================================

PAYLOAD_CASE=${PAYLOAD_CASE:-3p1} \
PAYLOAD_TORQUE_DERATE=${PAYLOAD_TORQUE_DERATE:-0.60} \
PAYLOAD_LOCK_SLIDE_YAW=0 \
PAYLOAD_CONSTRAINED_SPACE=atlas \
PAYLOAD_SOLVE_TIME=${PAYLOAD_SOLVE_TIME:-30} \
PAYLOAD_DELTA=0.08 \
PAYLOAD_BEST_OF_N=1 \
PAYLOAD_BUDGET=${PAYLOAD_BUDGET:-150} \
PAYLOAD_FLOOR_CLEARANCE=0.0 \
PAYLOAD_WALL=0 \
PAYLOAD_SURF_A_Z=0.00 \
PAYLOAD_SURF_A_XY="0.09 0.20 0.38 0.58" \
PAYLOAD_SURF_B_Z=${PAYLOAD_SURF_B_Z:-0.30} \
PAYLOAD_SURF_B_XY="0.09 -0.58 0.38 -0.20" \
python -B run_panda.py "$@"