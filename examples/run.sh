#!/usr/bin/env bash
# Example: QM/MM MD of 1il4 ligand (50 steps) with various MLIP backends.
# Usage: ./run.sh
set -e

PARM=data/leap.parm7
RST=data/md.rst7

# --- UMA ---
echo "===== UMA ====="
amber-mlips -O -i uma.in -o uma.out -p $PARM -c $RST -r uma.rst7

# --- ORB ---
echo "===== ORB ====="
amber-mlips -O -i orb.in -o orb.out -p $PARM -c $RST -r orb.rst7

# --- MACE ---
echo "===== MACE ====="
amber-mlips -O -i mace.in -o mace.out -p $PARM -c $RST -r mace.rst7

# --- AIMNet2 ---
echo "===== AIMNet2 ====="
amber-mlips -O -i aimnet2.in -o aimnet2.out -p $PARM -c $RST -r aimnet2.rst7

# --- UMA + embedcharge ---
echo "===== UMA + embedcharge ====="
amber-mlips -O -i uma_embedcharge.in -o uma_embedcharge.out -p $PARM -c $RST -r uma_embedcharge.rst7
