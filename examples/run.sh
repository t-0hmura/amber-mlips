#!/bin/bash
# Example: QM/MM MD of 1il4 ligand (50 steps) with various MLIP backends.

# --- UMA ---
amber-mlips --mm-ranks 16 -O -i uma.in -o uma.out -p leap.parm7 -c md.rst7 -r uma.rst7

# --- ORB ---
amber-mlips --mm-ranks 16 -O -i orb.in -o orb.out -p leap.parm7 -c md.rst7 -r orb.rst7

# --- MACE ---
amber-mlips --mm-ranks 16 -O -i mace.in -o mace.out -p leap.parm7 -c md.rst7 -r mace.rst7

# --- AIMNet2 ---
amber-mlips --mm-ranks 16 -O -i aimnet2.in -o aimnet2.out -p leap.parm7 -c md.rst7 -r aimnet2.rst7

# --- UMA + embedcharge ---
amber-mlips --mm-ranks 16 -O -i uma_embedcharge.in -o uma_embedcharge.out -p leap.parm7 -c md.rst7 -r uma_embedcharge.rst7

# --- ML-Only MD (implicit solvent, non-periodic) ---
amber-mlips -O -i uma_mlonly_implicit.in -o mlonly_implicit.out -p water.parm7 -c water.rst7 -r mlonly_implicit.rst7
