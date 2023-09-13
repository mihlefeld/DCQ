import os
import itertools

# alpha_modes = ["divide", "noop", "quantize"]
# cluster_modes = ["median_cuts", "k_means"]
# dither_modes = ["FS", "FS-DCQ"]
alpha_modes = ["divide", "noop"]
cluster_modes = ["_"]
dither_modes = ["DCQ"]
for am, cm, dm in itertools.product(alpha_modes, cluster_modes, dither_modes):
    cmd = f'python run_all.py -i _data/pictures -o "_data/qpn8/{dm} {cm} {am}" -p 8 -d ./cmake-build-release/DCQ -a {am} -c {cm} -m {dm}'
    print(f'python run_all.py -i _data/pictures -o "_data/qpn8/{dm} {cm} {am}" -p 8 -d ./cmake-build-release/DCQ -a {am} -c {cm} -m {dm}')
    os.system(cmd)
