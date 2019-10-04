#
# copy_checkpoints.py
#
# Run this script with specified source_dir and target_dir while the model is training to make a copy
# of every checkpoint (checkpoints are kept once an hour by default and is difficult to adjust)
#

import time
import os
import shutil

check_every_n_minutes = 10

source_dir = '/datadrive/megadetectorv3/experiments/190425'
target_dir = '/datadrive/megadetectorv3/experiments/0425_checkpoints'

os.makedirs(target_dir, exist_ok=True)


num_checks = 0

while True:
    
    num_checks += 1
    print('Checking round {}.'.format(num_checks))

    for f in os.listdir(source_dir):
        # do not copy event or evaluation results
        if f.startswith('model') or f.startswith('graph'):
            target_path = os.path.join(target_dir, f)
            if not os.path.exists(target_path):
                _ = shutil.copy(os.path.join(source_dir, f), target_path)
                print('Copied {}.'.format(f))

    print('End of round {}.'.format(num_checks))

    time.sleep(check_every_n_minutes * 60)
