import os


os.system('torchrun --nproc_per_node=1 ./opengait/main.py --cfgs ./config/stdn/casiab/stdn-ca.yaml --phase train --log_to_file')
# os.system('torchrun --nproc_per_node=1 ./opengait/main.py --cfgs ./config/gaitgl/gaitgl-335.yaml --phase test --log_to_file')

