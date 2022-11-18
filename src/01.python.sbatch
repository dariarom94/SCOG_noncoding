#!/bin/bash
#SBATCH --job-name episcanpy
#SBATCH --output ../log/noncoding_%A_%a.txt
#SBATCH --partition tinyq
#SBATCH --qos tinyq
#SBATCH --mem 32G
#SBATCH --array=1-9


export SCRATCH_PATH=/nobackup/lab_bock/users/dromanovskaia/mytmp
export TMPDIR=$SCRATCH_PATH/Rtmp


ANNOT_TABLE="/home/dromanovskaia/projects/SCOG_episcanpy/data/noncoding_bed_files/bedlist.txt"

sample=`awk -F"\t" 'NR=='$SLURM_ARRAY_TASK_ID'{print $1}' ${ANNOT_TABLE}`
echo $sample

date
/nobackup/lab_bock/users/dromanovskaia/miniconda3/envs/myepiscanpy/bin/python 01_testRegions.py -b ${sample} -t ${1}
date
