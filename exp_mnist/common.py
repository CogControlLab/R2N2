import pickle
import sys
import numpy as np
from pathlib import Path
root_path = Path('..')
sys.path.append(str(root_path.absolute()))

data_path = root_path / "data/mnist_binary.pkl"
with open(data_path, 'rb') as f:
    mnist = pickle.load(f)

x_train = mnist["training_images"]
y_train = mnist["training_labels"]

n_test_samples = 2000
indices_test = np.random.choice(
    len(mnist["test_images"]),
    n_test_samples,
    replace=False,
)
x_test = mnist["test_images"][indices_test]
y_test = mnist["test_labels"][indices_test]

# real settings
n_x = 28
T = n_x

n_h = 64
n_samples = x_train.shape[0]
n_epoch = 5

is_masked = True
mask_steps = 10
mask_value = 0

# for debugging
# n_samples = 10
