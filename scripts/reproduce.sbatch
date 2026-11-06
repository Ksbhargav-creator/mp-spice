#!/bin/bash
# Slurm array job: one task per README matrix, via the Apptainer image built from
# Dockerfile.reproduce. NOT yet run on a real cluster -- check the #SBATCH lines
# against your site's partitions/limits before submitting.
#
#   apptainer build mp-spice-repro.sif docker-daemon://mp-spice-repro:latest
#   sbatch scripts/reproduce.sbatch
#   # after the whole array has finished (or timed out):
#   apptainer run --bind $PWD/results:/results mp-spice-repro.sif merge
#
# Index -> matrix mapping:  apptainer run mp-spice-repro.sif list
#
# Resource guesses -- NOT measured. rajat30 (643,994 rows, 6.2M nonzeros) is far
# larger than every other matrix; if you want tight requests, give it its own
# submission (`sbatch --array=12 --mem=64G ...`) and leave the other 13 small.
#SBATCH --job-name=mp-spice-repro
#SBATCH --array=0-13
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%A_%a.out

set -euo pipefail

SIF="${SIF:-$PWD/mp-spice-repro.sif}"
RESULTS="${RESULTS:-$PWD/results}"
mkdir -p "$RESULTS"

# --index is explicit rather than relying on the environment crossing the container
# boundary (it would not survive `apptainer run --cleanenv`).
apptainer run --bind "$RESULTS":/results "$SIF" run --index "$SLURM_ARRAY_TASK_ID"
