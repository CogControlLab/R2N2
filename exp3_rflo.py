"""
Sequential MNIST for RFLO algorithm
"""
import numpy as np
import matplotlib.pyplot as plt
from models import R2N2, CtlRNN
from utility import h_seq_pre_processing, stable_softmax
from figures import show_activity_forward_backward, show_loss_curve, output_target_comparison, show_loss_and_accuracy
import ipdb
from tqdm import tqdm, trange
import numpy as np
import pickle
from copy import deepcopy
from pathlib import Path

from rflo_model import RfloRNN

data_path = "data/mnist_binary.pkl"
with open(data_path, 'rb') as f:
    mnist = pickle.load(f)

x_train = mnist["training_images"]
y_train = mnist["training_labels"]
x_test = mnist["test_images"]
y_test = mnist["test_labels"]

exp_name = "exp_5_RFLO"

n_x = 28
T = n_x

n_h = 64
n_h_t = n_h * 2  # we have two groups of neurons in R2N2 model
# lr = 1e-3
lr = 0.1  # same as the lr in original Rflo paper.

n_samples = x_train.shape[0]
n_epoch = 5

h_init = 0.1 * np.ones(n_h_t)

model = RfloRNN(n_x, n_h_t, 10, h_init)

kk = trange(n_epoch, desc="Echo State RNN: ", leave=True)
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
        y = int(y_train[i])

        prediction, loss = model.run_trial_seq_classification(
            xs,
            y,
            eta=[lr, lr, lr],
            learning='rflo',
            online_learning=False,
        )
        loss_list_t.append(loss)
        results_list_t.append(int(prediction == y))
        kk.set_description(
            f"Comparison Epoch {e+1} | Train: Acc: {np.around(np.mean(results_list_t[-1000:]), 2)} | L: {np.around(np.mean(loss_list_t[-1000:]), 2)}; Validate in epoch {e}: Acc {np.around(acc_v_t, 2)}"
        )

show_loss_and_accuracy(loss_list_t,
                       results_list_t,
                       exp_name=exp_name,
                       show=True,
                       yL_label="Crosss entropy",
                       yR_label="Accuracy")

pickle.dump({
    "loss": loss_list_t,
    "result": results_list_t
}, open(f"results/{exp_name}.pkl", "wb"))