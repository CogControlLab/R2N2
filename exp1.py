"""
Experiment 1: show the basic capability to remember some sequences.
"""
import numpy as np
import matplotlib.pyplot as plt
from models import R2N2
from utility import h_seq_pre_processing
from figures import show_activity_forward_backward, show_loss_curve, output_target_comparison
import ipdb
import pickle
from tqdm import tqdm

# np.random.seed(1926)


def mse_loss(output, target):
    return np.sum(0.5 * (output - target)**2)


def d_mse_loss(output, target):
    return output - target


exp_name = "exp_1"
n_x = 6
n_h = 64
lr = 5e-3
model = R2N2(n_h, n_x, g=2.5, tau=10, lr=lr, enableFA=True)
T = 20

n_trials = 50000

xs = np.random.rand(T, n_x) * 0.5

xs = np.zeros((T, n_x))
for t in range(T):
    # xs[t, t] = 1
    xs[t, np.random.choice(n_x, replace=False)] = 1

loss_list = []

for i in tqdm(range(n_trials)):
    model.zero_grads()
    h_forward_seq, output_seq, loss = model.forward_seq_prediction(
        xs, mse_loss)
    # this part shoud be replaced by incremental learning.
    model.hard_reset_w_backward()
    h_backward_seq = model.backward_learning_seq_prediction(
        h_forward_seq[-1], xs, d_mse_loss)

    model.apply_grads_Adam(T)

    loss_list.append(loss)

    if np.mean(loss_list[-1000:]) < 1e-3:
        break

show_loss_curve(loss_list,
                time_normalization=True,
                t_max=T,
                ylabel="MSE",
                exp_name=exp_name,
                show=True)

output_seq = np.array(output_seq)
output_target_comparison(
    output_seq,
    xs[1:],
    exp_name=exp_name,
)
pickle.dump({
    "output": output_seq,
    "input": xs[1:]
}, open(f"results/{exp_name}_comparison.pkl", "wb"))

# ipdb.set_trace()
hs_A_f, hs_B_f = h_seq_pre_processing(h_forward_seq)
hs_A_b, hs_B_b = h_seq_pre_processing(h_backward_seq)
show_activity_forward_backward(hs_A_f,
                               hs_B_f,
                               hs_A_b,
                               hs_B_b,
                               exp_name=exp_name)
