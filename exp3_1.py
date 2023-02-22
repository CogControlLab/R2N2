"""
Hyperparameter tuning for R2N2.
"""
import argparse
import optuna
import numpy as np

from models import R2N2
from utility import cross_entropy_loss, d_cross_entropy_loss
# import ipdb
# from tqdm import trange
import pickle

parser = argparse.ArgumentParser()
parser.add_argument(
    "--cutoff",
    help="Number of samples for training and validation",
    type=int,
    default=-1,
)
parser.add_argument(
    "--trials",
    help="Number of trials",
    type=int,
    default=10,
)
parser.add_argument(
    "--storage",
    help="storage",
    type=str,
    default="sqlite:///hyperparams-optimization/R2N2-MNIST.db",
)

parser.add_argument(
    "--study_name",
    help="Study name",
    type=str,
    default="R2N2-MNIST",
)

args = parser.parse_args()


def validate(model, x_v, y_v):
    results_list = []
    loss_list = []
    for i in range(x_v.shape[0]):
        xs = x_v[i].reshape(T, T)
        y = y_v[i]
        h_forward_seq, output_seq, loss = model.forward_seq_decision_making(
            xs, y, cross_entropy_loss)

        loss_list.append(loss)

        prediction = np.argmax(output_seq[-1])
        results_list.append(prediction == y)

    return np.mean(results_list), np.mean(loss_list)


data_path = "data/mnist_binary.pkl"
with open(data_path, 'rb') as f:
    mnist = pickle.load(f)

k_train = args.cutoff
k_test = args.cutoff
x_train = mnist["training_images"][:k_train]
y_train = mnist["training_labels"][:k_train]

x_test = mnist["test_images"][:k_test]
y_test = mnist["test_labels"][:k_test]

n_x = 28
T = n_x


def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    1e-3
    scaling_out_lr = trial.suggest_loguniform('scaling_out_lr', 1, 100)
    g_forward = trial.suggest_float("g_forward", 1.2, 2)
    g_backward = trial.suggest_float("g_backward", 0.8, 1.5)

    n_epoch = 3  # fixed
    n_h = 64  # fixed

    model = R2N2(
        n_h,
        10,
        n_input=28,
        g=2.5,
        tau=10,
        lr=lr,
        enableFA=True,
        g_forward=g_forward,
        g_backward=g_backward,
    )

    loss_list = []
    results_list = []

    n_samples = x_train.shape[0]

    # kk = trange(n_epoch, desc="Average Classification Accuracy: ", leave=True)
    kk = range(n_epoch)
    acc_v, loss_v = 0, 0
    acc_v_list = []

    for e in kk:
        acc_v, loss_v = validate(model, x_test, y_test)
        acc_v_list.append(acc_v)
        for i in range(n_samples):
            model.zero_grads()
            xs = x_train[i].reshape(T, T)
            y = y_train[i]
            h_forward_seq, output_seq, loss = model.forward_seq_decision_making(
                xs, y, cross_entropy_loss)

            model.hard_reset_w_backward()
            _ = model.backward_learning_decision_making(
                h_forward_seq[-1], xs, y, d_cross_entropy_loss)

            model.apply_grads_Adam(T, scaling_out_lr=scaling_out_lr)
            loss_list.append(loss)

            prediction = np.argmax(output_seq[-1])
            results_list.append(prediction == y)
            if (i + 1) % 10 == 0:
                s = f"Epoch {e+1} Step {i+1} | Train: Acc: {np.around(np.mean(results_list[-100:]), 2)} | L: {np.around(np.mean(loss_list[-100:]), 2)}; Validate in epoch {e}: Acc {np.around(acc_v, 2)}"
                # kk.set_description(s)
                print(s)

    # return np.mean(loss_list[-500:])
    return np.mean(results_list[-2000:])


study = optuna.create_study(
    study_name=args.study_name,
    storage=args.storage,
    load_if_exists=True,
    direction='maximize',
)

study.optimize(objective, n_trials=args.trials)
