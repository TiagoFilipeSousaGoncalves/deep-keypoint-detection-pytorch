#!/bin/bash
#
#SBATCH -p v100_32GB              # Partition
#SBATCH --qos=v100                # QOS
#SBATCH --job-name=cinderella_dkd      # Job name
#SBATCH -o slurm.%N.%j.out             # STDOUT
#SBATCH -e slurm.%N.%j.err             # STDERR



echo "CINDERELLA Deep Keypoint Detection | PICTURE-DB | Started"

python code/train.py --database picture-db --epochs 300 --batch_size 8 --num_workers 4 --gpu_id 0 --results_directory results

echo "CINDERELLA Deep Keypoint Detection | PICTURE-DB | Finished"
