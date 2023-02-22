import sys
from pathlib import Path
root_path = Path('..')
sys.path.append(str(root_path.absolute()))

# real settings
n_x = 3
n_h = 32
n_repeat = 10
n_epoch = 10000

# for debugging
# n_repeat = 2
# n_epoch = 5
