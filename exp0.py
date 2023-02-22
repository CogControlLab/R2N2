"""
Experiment 1: reconstruction of activation sequences without external signals
"""
import numpy as np
import matplotlib.pyplot as plt
from models import R2N2
from utility import h_seq_pre_processing
from figures import show_activity_forward_backward, show_loss_curve, output_target_comparison
import ipdb
from tqdm import tqdm
import pickle
# np.random.seed(1926)


def mse_loss(output, target):
    return np.sum(0.5 * (output - target)**2)


def d_mse_loss(output, target):
    return output - target


exp_name = "exp_0"
n_x = 10
n_h = 64
lr = 5e-3
model = R2N2(n_h, n_x, g=2.5, tau=10, lr=lr, enableFA=True)
T = 100
for c in range(4):
    xs = np.random.rand(T, n_x) * 0.5
    h_forward_seq, output_seq = model.forward(xs)
    model.solve_w_backward()

h_backward_seq = model.backward_replay(h_forward_seq[-1], xs)

hs_A_f, hs_B_f = h_seq_pre_processing(h_forward_seq)
hs_A_b, hs_B_b = h_seq_pre_processing(h_backward_seq)
show_activity_forward_backward(hs_A_f,
                               hs_B_f,
                               hs_A_b,
                               hs_B_b,
                               exp_name=exp_name)

pickle.dump(
    {
        "hs": {
            "hs_A_f": hs_A_f,
            "hs_B_f": hs_B_f,
            "hs_A_b": hs_A_b,
            "hs_B_b": hs_B_b,
        }
    }, open(f"results/{exp_name}.pkl", "wb"))
