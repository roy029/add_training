#!/bin/bash


#$ -l rt_G.small=1
#$ -l h_rt=24:00:00
#$-j y
#$-m b
#$-m a
#$-m e
#$-cwd


source /etc/profile.d/modules.sh
module load python/3.6/3.6.5
module load cuda/9.2/9.2.88.1

#!/bin/bash -e
pip3 install --user --upgrade pip
pip3 install -r requirements.txt
pip3 install slackweb
python3 mask_T5.py conala-mined.txt
python3 slack.py

#!/bin/bash
# ./a.out