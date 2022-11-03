{\rtf1\ansi\ansicpg1252\cocoartf2639
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;\f1\fnil\fcharset0 HelveticaNeue;\f2\fnil\fcharset0 Menlo-Regular;
}
{\colortbl;\red255\green255\blue255;\red38\green38\blue38;\red37\green37\blue37;\red255\green255\blue255;
\red32\green108\blue135;\red0\green0\blue0;\red32\green108\blue135;}
{\*\expandedcolortbl;;\cssrgb\c20000\c20000\c20000;\cssrgb\c19216\c19216\c19216;\cssrgb\c100000\c100000\c100000;
\cssrgb\c14902\c49804\c60000;\cssrgb\c0\c0\c0;\cssrgb\c14902\c49804\c60000;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\fs34 \cf2 \expnd0\expndtw0\kerning0
#!/bin/sh\
\
\pard\pardeftab720\partightenfactor0

\f1\fs24\fsmilli12150 \cf3 \cb4 #SBATCH -n 10 # Request number of cores\
#SBATCH -N 1 # Request 1 node\
#SBATCH -t 12:00:00 # Time limit hrs:min:sec\
#SBATCH --mem-per-cpu=32000 #Request 2G of memory per CPU\
#SBATCH -o latest_output.txt #redirect output\cf3 \cb1 \
\cf3 \cb4 #SBATCH -e latest_error.txt #redirect errors\
\
module load anaconda3/2021.11\
pip3 install 
\f2\fs24 \cf5 \cb4 \outl0\strokewidth0 \strokec5 BeautifulSoup\cf0 \cb1 \strokec6 \
\pard\pardeftab720\partightenfactor0

\f1\fs24\fsmilli12150 \cf3 \cb4 \outl0\strokewidth0 pip3 install 
\f2\fs24 \cf7 tqdm\
pip3 install re\
pip3 install nltk\
pip3 install langid\
\
\pard\pardeftab720\partightenfactor0

\f1\fs24\fsmilli12150 \cf3 srun python3 run_cleaning.py}