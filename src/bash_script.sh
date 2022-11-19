#!/bin/sh

#SBATCH -N 1 # Request 1 node
#SBATCH -t 12:00:00 # Time limit hrs:min:sec
#SBATCH --mem-per-cpu=32000 #Request 2G of memory per CPU
#SBATCH -o latest_output.txt #redirect output
#SBATCH -e latest_error.txt #redirect errors

module load anaconda3/2021.11
pip3 install BeautifulSoup
pip3 install tqdm
pip3 install re
pip3 install nltk
pip3 install langid

srun python3 run_cleaning.py
