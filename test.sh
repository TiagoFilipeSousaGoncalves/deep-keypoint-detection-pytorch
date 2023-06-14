#!/bin/bash
#
#SBATCH -p gtx1080_8GB                  # Partition
#SBATCH --qos=gtx1080                   # QOS
#SBATCH --job-name=cinderella_dkd       # Job name
#SBATCH -o slurm.%N.%j.out              # STDOUT
#SBATCH -e slurm.%N.%j.err              # STDERR



echo "CINDERELLA Deep Keypoint Detection | PICTURE-DB | Started"

python code/test.py --database picture-db --gpu_id 0 --model_directory /nas-ctm01/homes/tgoncalv/deep-keypoint-detection-pytorch/results/2023-06-13_23-06-50/

echo "CINDERELLA Deep Keypoint Detection | PICTURE-DB | Finished"
