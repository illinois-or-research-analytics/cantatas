#!/bin/bash
#SBATCH --job-name=cantatas
#SBATCH --account=chackoge-ic
#SBATCH --partition=secondary
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=cantatas_%j.out
#
# One link of a long CantataS run. Copy this file into the run directory,
# adjust the #SBATCH lines and the variables below, and submit it once:
#
#     sbatch run_chained.sbatch
#
# Each link resumes from the newest snapshot in checkpoint_dir (or starts the
# run if there is none), pauses before the job's time limit, writes a
# snapshot, and submits the next link. The chain stops when the simulation
# finishes (exit 0) or fails (any exit code other than 3).
#
# Keep checkpoint_dir on /scratch or /projects: node-local storage is wiped
# when the job ends. Set num_processors in the config to --cpus-per-task.

CONFIG=${CONFIG:-$SLURM_SUBMIT_DIR/config.ini}
ABM=${ABM:-/path/to/build/bin/abm}
SCRIPT=${SCRIPT:-$SLURM_SUBMIT_DIR/run_chained.sbatch}

module purge
module load gcc/13.3.0 openmpi/5.0.1-gcc-13.3.0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
# The simulator reads SLURM_JOB_END_TIME; pass the end time explicitly as
# well in case the scheduler did not set it.
if [ -z "${SLURM_JOB_END_TIME:-}" ]; then
  export CANTATAS_DEADLINE_EPOCH=$(date -d "$(squeue -h -j "$SLURM_JOB_ID" -o %e)" +%s)
fi

mpirun -n "$SLURM_NNODES" --map-by ppr:1:node --bind-to none \
  -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES -x CANTATAS_DEADLINE_EPOCH \
  "$ABM" --config "$CONFIG" --resume auto
status=$?

case $status in
  0) echo "Simulation finished." ;;
  3) echo "Paused with a snapshot; submitting the next link."
     sbatch --export=ALL,CONFIG="$CONFIG",ABM="$ABM",SCRIPT="$SCRIPT" "$SCRIPT" ;;
  *) echo "abm failed with exit code $status; not resubmitting." ;;
esac
exit $status
