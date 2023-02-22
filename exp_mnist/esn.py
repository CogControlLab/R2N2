"""
Experiment 3:
Compare the capability of R2N2 and classical RNN in sequential hand written didigt classification task.

This script is for ESN (Echo State Network)

Data is from https://github.com/aiddun/binary-mnist.
"""
import sys
from pathlib import Path
root_path = Path('..')
sys.path.append(str(root_path.absolute()))
import numpy as np

from figures import show_loss_and_accuracy
# import ipdb
from tqdm import tqdm, trange

import pickle

from pathlib import Path

from rflo_model import RfloRNN
from common import x_train, y_train, x_test, y_test, n_h, n_epoch, n_samples, T, is_masked, mask_steps

exp_name = f"MNIST_ESN-masked-{is_masked}-{mask_steps}"

n_x = 28

n_h_t = n_h * 2  # we have two groups of neurons in R2N2 model
# lr = 1e-3
lr = 0.1  # same as the lr in original Rflo paper.

h_init = 0.1 * np.ones(n_h_t)

model = RfloRNN(n_x, n_h_t, 10, h_init)

kk = trange(n_epoch, desc="ESN", leave=True)
loss_list_t = []
results_list_t = []

acc_v_t, loss_v_t = 0, 0
acc_v_list_t = []


def validate_t(x_v, y_v):
    n_samples = x_v.shape[0]
    loss_list_t_v = []
    results_list_t_v = []

    for i in range(n_samples):
        xs = x_v[i].reshape(T, -1)
        if is_masked:
            xs = xs[mask_steps:, :]
        y = int(y_v[i])

        prediction, loss = model.run_trial_seq_classification(
            xs,
            y,
            eta=[0, 0, 0],
            learning='rflo',
            online_learning=False,
        )
        loss_list_t_v.append(loss)
        results_list_t_v.append(int(prediction == y))

    return np.mean(results_list_t_v), np.mean(loss_list_t_v)


for e in kk:
    acc_v_t, loss_v_t = validate_t(x_test[:1000], y_test[:1000])
    acc_v_list_t.append(acc_v_t)
    for i in tqdm(range(n_samples)):
        xs = x_train[i].reshape(T, -1)
        if is_masked:
            xs = xs[mask_steps:, :]
        y = int(y_train[i])

        prediction, loss = model.run_trial_seq_classification(
            xs,
            y,
            eta=[lr, 0, lr],
            learning='rflo',
            online_learning=False,
        )
        loss_list_t.append(loss)
        results_list_t.append(int(prediction == y))
        kk.set_description(
            f"ESN | Epoch {e+1} | Train: Acc: {np.around(np.mean(results_list_t[-1000:]), 2)} | L: {np.around(np.mean(loss_list_t[-1000:]), 2)}; Validate in epoch {e}: Acc {np.around(acc_v_t, 2)}"
        )

show_loss_and_accuracy(loss_list_t,
                       results_list_t,
                       exp_name=exp_name,
                       fig_path=root_path / f"figs/exp_mnist/{exp_name}.png",
                       show=True,
                       yL_label="Crosss entropy",
                       yR_label="Accuracy")

pickle.dump({
    "loss": loss_list_t,
    "result": results_list_t
}, open(root_path / f"results/exp_mnist/{exp_name}.pkl", "wb"))
