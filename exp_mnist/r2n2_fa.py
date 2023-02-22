"""
Experiment 3:
Compare the capability of R2N2 and classical RNN in sequential hand written didigt classification task.

This script is for R2N2 with FA

Data is from https://github.com/aiddun/binary-mnist.
"""
import sys
from pathlib import Path
root_path = Path('..')
sys.path.append(str(root_path.absolute()))
import numpy as np

from utility import cross_entropy_loss, d_cross_entropy_loss
from figures import show_loss_and_accuracy
# import ipdb
from tqdm import tqdm, trange
import pickle

from models import R2N2
from common import x_train, y_train, x_test, y_test, n_h, n_epoch, n_samples, T, is_masked, mask_steps

exp_name = f"MNIST_R2N2_FA-masked-{is_masked}-{mask_steps}"

lr = 5e-4

model = R2N2(n_h, 10, n_input=28, g=2.5, tau=10, lr=lr, enableFA=True)

loss_list = []
results_list = []


def validate(x_v, y_v):
    results_list = []
    loss_list = []
    for i in range(x_v.shape[0]):
        xs = x_v[i].reshape(T, T)
        if is_masked:
            xs = xs[mask_steps:, :]
        y = y_v[i]
        h_forward_seq, output_seq, loss = model.forward_seq_decision_making(
            xs, y, cross_entropy_loss)

        loss_list.append(loss)

        prediction = np.argmax(output_seq[-1])
        results_list.append(prediction == y)

    return np.mean(results_list), np.mean(loss_list)


kk = trange(n_epoch, desc="R2N2_FA", leave=True)
acc_v, loss_v = 0, 0
acc_v_list = []

for e in kk:
    acc_v, loss_v = validate(x_test, y_test)
    acc_v_list.append(acc_v)
    for i in tqdm(range(n_samples)):
        model.zero_grads()
        xs = x_train[i].reshape(T, T)
        if is_masked:
            xs = xs[mask_steps:, :]
        y = y_train[i]
        h_forward_seq, output_seq, loss = model.forward_seq_decision_making(
            xs, y, cross_entropy_loss)

        model.hard_reset_w_backward()
        h_backward_seq = model.backward_learning_decision_making(
            h_forward_seq[-1], xs, y, d_cross_entropy_loss)

        model.apply_grads_Adam(1)
        loss_list.append(loss)

        prediction = np.argmax(output_seq[-1])
        results_list.append(prediction == y)
        # print(prediction, y)

        kk.set_description(
            f"R2N2_FA | Epoch {e+1} | Train: Acc: {np.around(np.mean(results_list[-1000:]), 2)} | L: {np.around(np.mean(loss_list[-1000:]), 2)}; Validate in epoch {e}: Acc {np.around(acc_v, 2)}"
        )

show_loss_and_accuracy(loss_list,
                       results_list,
                       exp_name=exp_name,
                       fig_path=root_path / f"figs/exp_mnist/{exp_name}.png",
                       show=True,
                       yL_label="Crosss entropy",
                       yR_label="Accuracy")

pickle.dump({
    "loss": loss_list,
    "result": results_list
}, open(root_path / f"results/exp_mnist/{exp_name}.pkl", "wb"))
