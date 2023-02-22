"""
Experiment 3:
Compare the capability of R2N2 and classical RNN in sequential hand written didigt classification task.

Data is from https://github.com/aiddun/binary-mnist.
"""
import numpy as np
import matplotlib.pyplot as plt
from models import R2N2, CtlRNN
from utility import h_seq_pre_processing, stable_softmax
from figures import show_activity_forward_backward, show_loss_curve, output_target_comparison, show_loss_and_accuracy
import ipdb
from tqdm import tqdm, trange
import numpy as np
from tasks import TMazeNavTask
import pickle
from copy import deepcopy
from pathlib import Path
import torch
import torch.nn as nn

data_path = "data/mnist_binary.pkl"
with open(data_path, 'rb') as f:
    mnist = pickle.load(f)

x_train = mnist["training_images"]
y_train = mnist["training_labels"]
x_test = mnist["test_images"]
y_test = mnist["test_labels"]

exp_name = "exp_5_EchoState"

n_x = 28
T = n_x

n_h = 64
lr = 5e-4

n_samples = x_train.shape[0]
n_epoch = 5

################################################################################################################
################################## EchoState RNN comparison ######################################################
################################################################################################################

x_train_t = torch.from_numpy(x_train).type(torch.FloatTensor)
y_train_t = torch.from_numpy(y_train).type(torch.LongTensor)
x_test_t = torch.from_numpy(x_test).type(torch.FloatTensor)
y_test_t = torch.from_numpy(y_test).type(torch.LongTensor)

loss_list_t = []
results_list_t = []

n_h_t = 2 * n_h

rnn_t = CtlRNN(n_h_t, 10, 28, tau=10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn_t.parameters(), lr=lr)

kk = trange(n_epoch, desc="Echo State RNN: ", leave=True)

acc_v_t, loss_v_t = 0, 0
acc_v_list_t = []


def validate_t(x_v, y_v):
    n_samples = x_v.shape[0]
    loss_list_t_v = []
    results_list_t_v = []
    # for i in tqdm(range(n_samples)):
    for i in range(n_samples):
        optimizer.zero_grad()
        n_batch = 1
        xs = x_v[i].reshape(T, n_batch, T)
        y = y_v[i].reshape(-1)
        h = torch.rand(n_batch, n_h_t)
        for t in range(T):
            h, o = rnn_t(h, xs[t])

        loss = criterion(o, y)
        loss.backward()
        optimizer.step()
        loss_list_t_v.append(loss.item())
        results_list_t_v.append((torch.argmax(o, dim=1) == y).item())

    return np.mean(results_list_t_v), np.mean(loss_list_t_v)


for e in kk:
    acc_v_t, loss_v_t = validate_t(x_test_t[:1000], y_test_t[:1000])
    acc_v_list_t.append(acc_v_t)
    for i in tqdm(range(n_samples)):
        optimizer.zero_grad()
        n_batch = 1
        xs = x_train_t[i].reshape(T, n_batch, T)
        y = y_train_t[i].reshape(-1)
        h = torch.rand(n_batch, n_h_t)
        for t in range(T):
            h, o = rnn_t(h, xs[t])
        # ipdb.set_trace()
        loss = criterion(o, y)
        loss.backward()
        optimizer.step()
        loss_list_t.append(loss.item())
        results_list_t.append((torch.argmax(o, dim=1) == y).item())
        kk.set_description(
            f"Comparison Epoch {e+1} | Train: Acc: {np.around(np.mean(results_list_t[-1000:]), 2)} | L: {np.around(np.mean(loss_list_t[-1000:]), 2)}; Validate in epoch {e}: Acc {np.around(acc_v_list_t, 2)}"
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