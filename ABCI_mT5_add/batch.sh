#!/bin/bash

#$-l rt_F=5
#$-j y
#$-cwd

source /etc/profile.d/modules.sh
module load python/3.6/3.6.5
module load cuda/9.2/9.2.88.1

#!/bin/bash -e
pip3 install --user --upgrade pip
pip3 install -r requirements.txt
python3 mask_T5.py output_small.txt

#!/bin/bash
# ./a.out